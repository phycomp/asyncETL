import asyncio
import psycopg
import psycopg_pool
from psycopg_pool import AsyncConnectionPool
from datetime import datetime, timedelta
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import json
import aiohttp
import aiometer
import time
import gc
from dataclasses import dataclass
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
from memory_profiler import profile
import signal
from functools import partial

# 設定記憶體使用監控
import resource
def memory_limit(percentage: float):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (
        int(get_memory() * 1024 * percentage),
        hard
    ))

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemTotal:'):
                free_memory = int(sline[1])
                break
    return free_memory

# 設定 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingMetrics:
    start_time: datetime
    records_processed: int = 0
    errors_count: int = 0
    retry_count: int = 0
    batch_times: List[float] = None
    
    def __post_init__(self):
        self.batch_times = []

    @property
    def average_batch_time(self) -> float:
        return sum(self.batch_times) / len(self.batch_times) if self.batch_times else 0

    @property
    def processing_duration(self) -> timedelta:
        return datetime.now() - self.start_time

    def to_dict(self) -> Dict:
        return {
            'total_records': self.records_processed,
            'errors': self.errors_count,
            'retries': self.retry_count,
            'avg_batch_time': self.average_batch_time,
            'total_duration': str(self.processing_duration),
            'records_per_second': self.records_processed / self.processing_duration.total_seconds()
            if self.processing_duration.total_seconds() > 0 else 0
        }

class AsyncBatchProcessor:
    def __init__(
        self,
        source_dsn: str,
        target_dsn: str,
        batch_size: int = 10000,
        max_concurrent_batches: int = 5,
        retry_attempts: int = 3,
        memory_limit_percentage: float = 0.8
    ):
        self.source_pool = AsyncConnectionPool(
            source_dsn,
            min_size=5,
            max_size=20
        )
        self.target_pool = AsyncConnectionPool(
            target_dsn,
            min_size=5,
            max_size=20
        )
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.retry_attempts = retry_attempts
        self.metrics = ProcessingMetrics(start_time=datetime.now())
        self.checkpoint_data = {}
        
        # 設定記憶體限制
        memory_limit(memory_limit_percentage)
        
        # 初始化同步原語
        self._batch_semaphore = asyncio.Semaphore(max_concurrent_batches)
        self._checkpoint_lock = asyncio.Lock()

    async def _execute_with_retry(self, func, *args, **kwargs) -> Any:
        """使用重試機制執行資料庫操作"""
        for attempt in range(self.retry_attempts):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == self.retry_attempts - 1:
                    raise
                self.metrics.retry_count += 1
                wait_time = 2 ** attempt
                logger.warning(f"操作失敗，{wait_time} 秒後重試: {str(e)}")
                await asyncio.sleep(wait_time)

    async def _save_checkpoint(self, batch_id: int, status: str):
        """儲存檢查點資訊"""
        async with self._checkpoint_lock:
            self.checkpoint_data[batch_id] = {
                'status': status,
                'timestamp': datetime.now().isoformat()
            }
            # 寫入檢查點檔案
            with open('etl_checkpoint.json', 'w') as f:
                json.dump(self.checkpoint_data, f)

    @asynccontextmanager
    async def _get_connection(self, pool):
        """智能連線管理"""
        conn = await pool.getconn()
        try:
            yield conn
        finally:
            await pool.putconn(conn)

    async def extract_batch(self, query: str, batch_number: int) -> pd.DataFrame:
        """使用 pandas 提取批次資料"""
        offset = batch_number * self.batch_size
        paginated_query = f"""
            {query}
            LIMIT {self.batch_size}
            OFFSET {offset}
        """
        
        async with self._get_connection(self.source_pool) as conn:
            async with conn.cursor() as cur:
                await cur.execute(paginated_query)
                columns = [desc[0] for desc in cur.description]
                data = await cur.fetchall()
                
                # 使用 pandas 處理資料
                df = pd.DataFrame(data, columns=columns)
                return df

    async def transform_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """使用 pandas 進行資料轉換"""
        # 這裡示範一些常見的轉換操作
        try:
            # 移除重複資料
            df = df.drop_duplicates()
            
            # 處理遺漏值
            df = df.fillna({
                'numeric_column': 0,
                'string_column': 'unknown'
            })
            
            # 資料型別轉換
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass
            
            # 新增衍生欄位
            if 'created_at' in df.columns:
                df['processing_date'] = datetime.now()
                df['age_days'] = (datetime.now() - df['created_at']).dt.days
            
            return df
            
        except Exception as e:
            logger.error(f"資料轉換錯誤: {str(e)}")
            raise

    async def load_batch(self, df: pd.DataFrame, table_name: str):
        """使用 COPY 命令大量載入資料"""
        if df.empty:
            return 0
            
        async with self._get_connection(self.target_pool) as conn:
            async with conn.cursor() as cur:
                # 準備 COPY 命令
                columns = df.columns.tolist()
                copy_sql = f"COPY {table_name} ({','.join(columns)}) FROM STDIN WITH CSV"
                
                # 將 DataFrame 轉換為 CSV 格式
                csv_data = df.to_csv(index=False, header=False)
                
                # 使用 COPY 命令載入資料
                await cur.copy_expert(copy_sql, csv_data)
                await conn.commit()
                
                return len(df)

    async def process_batch(self, query: str, table_name: str, batch_number: int):
        """處理單一批次"""
        async with self._batch_semaphore:
            start_time = time.time()
            try:
                # 檢查是否已處理過此批次
                if str(batch_number) in self.checkpoint_data:
                    if self.checkpoint_data[str(batch_number)]['status'] == 'completed':
                        logger.info(f"批次 {batch_number} 已處理完成，跳過")
                        return 0

                # 提取資料
                df = await self._execute_with_retry(
                    self.extract_batch,
                    query,
                    batch_number
                )
                
                if df.empty:
                    return 0

                # 轉換資料
                df = await self._execute_with_retry(
                    self.transform_batch,
                    df
                )

                # 載入資料
                processed_count = await self._execute_with_retry(
                    self.load_batch,
                    df,
                    table_name
                )

                # 更新指標
                self.metrics.records_processed += processed_count
                self.metrics.batch_times.append(time.time() - start_time)
                
                # 儲存檢查點
                await self._save_checkpoint(batch_number, 'completed')
                
                # 強制記憶體回收
                del df
                gc.collect()

                return processed_count

            except Exception as e:
                self.metrics.errors_count += 1
                logger.error(f"處理批次 {batch_number} 時發生錯誤: {str(e)}")
                await self._save_checkpoint(batch_number, 'failed')
                raise

    async def run_etl(self, extract_query: str, target_table: str):
        """執行完整的 ETL 流程"""
        try:
            batch_number = 0
            tasks = []
            
            while True:
                # 建立批次處理任務
                task = asyncio.create_task(
                    self.process_batch(extract_query, target_table, batch_number)
                )
                tasks.append(task)
                
                # 當達到最大併發數時，等待部分任務完成
                if len(tasks) >= self.max_concurrent_batches * 2:
                    done, tasks = await asyncio.wait(
                        tasks,
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    # 檢查完成的任務是否有錯誤
                    for completed_task in done:
                        if completed_task.exception():
                            raise completed_task.exception()
                        
                        if completed_task.result() == 0:
                            # 沒有更多資料要處理
                            if tasks:
                                # 等待剩餘任務完成
                                await asyncio.gather(*tasks)
                            return
                
                batch_number += 1
                
                # 定期輸出處理進度
                if batch_number % 10 == 0:
                    logger.info(f"ETL 進度: {self.metrics.to_dict()}")

        finally:
            # 關閉連線池
            await self.source_pool.close()
            await self.target_pool.close()
            
            # 輸出最終統計資料
            logger.info("ETL 完成，最終統計：")
            logger.info(json.dumps(self.metrics.to_dict(), indent=2))

# 使用範例
async def main():
    # 資料庫連線設定
    source_dsn = "postgresql://user:password@source_host:5432/source_db"
    target_dsn = "postgresql://user:password@target_host:5432/target_db"
    
    # 初始化處理器
    processor = AsyncBatchProcessor(
        source_dsn=source_dsn,
        target_dsn=target_dsn,
        batch_size=10000,
        max_concurrent_batches=5,
        retry_attempts=3,
        memory_limit_percentage=0.8
    )
    
    # 定義提取查詢
    extract_query = """
        SELECT 
            id,
            name,
            email,
            created_at,
            updated_at,
            status,
            data::jsonb as json_data
        FROM large_table
        WHERE updated_at >= current_date - interval '7 days'
        ORDER BY id
    """
    
    # 註冊信號處理器以優雅地處理中斷
    def signal_handler():
        logger.info("收到中斷信號，正在優雅地關閉...")
        # 儲存當前狀態
        with open('etl_state.json', 'w') as f:
            json.dump(processor.metrics.to_dict(), f)
        
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, signal_handler)
    
    # 執行 ETL
    await processor.run_etl(extract_query, "processed_data")

if __name__ == "__main__":
    asyncio.run(main())
