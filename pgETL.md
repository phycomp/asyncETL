使用 psycopg async異步 進階範例ETL
使用 psycopg 的異步功能來實現 ETL（Extract, Transform, Load）流程，能夠有效提高對數據庫的高效訪問和處理能力，特別是處理大量數據時，異步操作可以充分利用系統資源。本範例將展示如何使用 psycopg 的異步接口從 PostgreSQL 提取數據、進行轉換並將結果加載到另一個表中。
1. 安裝 psycopg
首先，確保安裝了 psycopg 以及支持異步操作的額外包：
pip install psycopg[async]
2. 定義異步 ETL 流程框架
這裡展示一個完整的異步 ETL 流程，包括數據的提取（Extract）、轉換（Transform）和加載（Load）。
步驟 1：設置異步連接和查詢
import asyncio
import psycopg
from psycopg.rows import dict_row

# 異步連接 Postgres 數據庫
async def connect_to_db():
    conn = await psycopg.AsyncConnection.connect(
        "dbname=test user=postgres password=yourpassword host=localhost"
    )
    return conn

# 提取數據（Extract）
async def extract_data(conn, query):
    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(query)
        data = await cur.fetchall()
    return data

# 轉換數據（Transform）
def transform_data(data):
    # 這裡可以根據需要進行數據轉換
    # 比如將某些欄位進行計算、格式轉換等
    transformed_data = []
    for row in data:
        # 假設我們將價格提升 10%
        row['price'] = row['price'] * 1.10
        transformed_data.append(row)
    return transformed_data

# 加載數據（Load）
async def load_data(conn, table_name, data):
    async with conn.cursor() as cur:
        # 插入新數據
        insert_query = f"INSERT INTO {table_name} (id, name, price) VALUES (%s, %s, %s)"
        for row in data:
            await cur.execute(insert_query, (row['id'], row['name'], row['price']))
        await conn.commit()

# ETL 流程
async def etl_process():
    # 1. 連接數據庫
    conn = await connect_to_db()

    try:
        # 2. 提取數據
        extract_query = "SELECT id, name, price FROM products"
        extracted_data = await extract_data(conn, extract_query)
        print(f"Extracted {len(extracted_data)} records")

        # 3. 轉換數據
        transformed_data = transform_data(extracted_data)
        print("Data transformation completed")

        # 4. 加載數據到目標表
        await load_data(conn, "products_transformed", transformed_data)
        print("Data loading completed")
        
    finally:
        # 確保關閉連接
        await conn.close()

# 運行異步 ETL 流程
asyncio.run(etl_process())
3. 各部分詳解
1. 異步連接數據庫
我們使用 psycopg.AsyncConnection.connect 方法來建立異步數據庫連接。這樣的連接允許我們在同一時間內處理多個 SQL 查詢，而不會阻塞主線程，這在高並發的情況下尤其有效。
python
複製程式碼
async def connect_to_db():
    conn = await psycopg.AsyncConnection.connect(
        "dbname=test user=postgres password=yourpassword host=localhost"
    )
    return conn
2. 提取數據（Extract）
使用 async with conn.cursor() 創建異步的游標（cursor），並執行查詢以提取數據。我們使用 row_factory=dict_row 將結果轉換為字典格式，以便更方便地對行數據進行操作。
python
複製程式碼
async def extract_data(conn, query):
    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(query)
        data = await cur.fetchall()
    return data
3. 轉換數據（Transform）
在轉換階段，根據需求對提取到的數據進行操作，例如進行計算、重新格式化等。在這個範例中，我們將 price 欄位提升 10%。
python
複製程式碼
def transform_data(data):
    transformed_data = []
    for row in data:
        row['price'] = row['price'] * 1.10  # 將價格提升 10%
        transformed_data.append(row)
    return transformed_data
4. 加載數據（Load）
異步地將轉換後的數據插入到目標表中，這裡我們假設插入的是 products_transformed 表。通過異步操作，插入數據可以大大加速。
python
複製程式碼
async def load_data(conn, table_name, data):
    async with conn.cursor() as cur:
        insert_query = f"INSERT INTO {table_name} (id, name, price) VALUES (%s, %s, %s)"
        for row in data:
            await cur.execute(insert_query, (row['id'], row['name'], row['price']))
        await conn.commit()
4. 進階應用與優化
1. 批量插入
在大量數據加載過程中，逐行插入的效率可能較低。可以使用批量插入來優化此過程，將多行數據一次性插入。
python
複製程式碼
async def load_data_batch(conn, table_name, data):
    async with conn.cursor() as cur:
        insert_query = f"INSERT INTO {table_name} (id, name, price) VALUES %s"
        values = [(row['id'], row['name'], row['price']) for row in data]
        await cur.executemany(insert_query, values)
        await conn.commit()
2. 並行數據提取和處理
可以使用 asyncio.gather 來並行執行多個提取或處理任務，進一步提高 ETL 流程的效率。例如，可以同時提取來自不同數據表的數據。
python
複製程式碼
async def etl_process():
    conn = await connect_to_db()

    try:
        extract_query1 = "SELECT id, name, price FROM products"
        extract_query2 = "SELECT id, category, price FROM categories"

        # 並行提取數據
        extracted_data1, extracted_data2 = await asyncio.gather(
            extract_data(conn, extract_query1),
            extract_data(conn, extract_query2)
        )

        # 轉換和加載數據的步驟與之前類似
        transformed_data1 = transform_data(extracted_data1)
        transformed_data2 = transform_data(extracted_data2)

        await load_data(conn, "products_transformed", transformed_data1)
        await load_data(conn, "categories_transformed", transformed_data2)

    finally:
        await conn.close()
5. 小結
這個 psycopg 異步 ETL 的進階範例展示了如何通過異步操作來提升數據提取、轉換和加載的效率。關鍵技術包括：
    • 使用 psycopg 的異步連接和游標進行高效的數據庫查詢。
    • 利用異步函數來進行非阻塞的 I/O 操作。
    • 結合 asyncio.gather 實現並行的數據提取和處理。
這樣的異步 ETL 流程適合用於處理大規模數據，並且可以根據具體業務需求進一步優化和擴展。
資料庫 async function 要如何存入 streamlit cache_data
在 Streamlit 中，我們可以使用 @st.cache_data 裝飾器來緩存資料，這樣可以避免多次重複請求相同的數據，提升應用程序的效率。然而當我們使用 async 函數與 Streamlit 進行數據庫交互時，直接將 async 函數與 cache_data 一起使用可能會導致問題，因為 @st.cache_data 目前不支援異步函數。
要解決這個問題，以下是幾種可能的方式：
方法 1：將異步函數結果轉換為同步函數並使用 cache_data
可以在 Streamlit 中使用 asyncio.run() 來將異步函數的結果轉換為同步，然後使用 @st.cache_data 來緩存結果。
範例
import streamlit as st
import asyncio
import psycopg
from psycopg.rows import dict_row

# 異步提取數據
async def async_extract_data():
    conn = await psycopg.AsyncConnection.connect(
        "dbname=test user=postgres password=yourpassword host=localhost"
    )
    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute("SELECT id, name, price FROM products")
        data = await cur.fetchall()
    await conn.close()
    return data

# 同步包裝器，使用 cache_data 裝飾器
@st.cache_data
def get_cached_data():
    # 在 Streamlit 中將異步函數轉換為同步函數
    return asyncio.run(async_extract_data())

# 使用 Streamlit 獲取並展示數據
def main():
    st.title("Async Database ETL with Streamlit Cache")
    
    # 使用緩存的數據
    data = get_cached_data()
    
    # 在 Streamlit 應用中顯示數據
    if data:
        st.write(f"Extracted {len(data)} records:")
        st.table(data)

if __name__ == "__main__":
    main()
方法 2：手動實現緩存機制
如果你需要完全異步化，且 Streamlit 的 @cache_data 不支援異步函數，那麼你可以自己實現簡單的緩存機制，將異步函數的結果手動緩存到變數或文件中。
範例
import streamlit as st
import asyncio
import psycopg
from psycopg.rows import dict_row
import time

# 手動實現一個簡單的緩存變數
cache = {"data": None, "timestamp": None}
cache_timeout = 60  # 緩存超時設定（秒）

# 異步提取數據
async def async_extract_data():
    conn = await psycopg.AsyncConnection.connect(
        "dbname=test user=postgres password=yourpassword host=localhost"
    )
    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute("SELECT id, name, price FROM products")
        data = await cur.fetchall()
    await conn.close()
    return data

# 檢查並更新緩存的數據
async def get_data_with_manual_cache():
    current_time = time.time()
    
    # 如果緩存過期或尚未緩存，重新提取數據
    if cache["data"] is None or (current_time - cache["timestamp"] > cache_timeout):
        print("Cache expired or empty, fetching new data...")
        data = await async_extract_data()
        cache["data"] = data
        cache["timestamp"] = current_time
    else:
        print("Using cached data...")
    
    return cache["data"]

# 使用 Streamlit 獲取並展示數據
def main():
    st.title("Async Database ETL with Manual Cache")

    # 使用異步任務
    data = asyncio.run(get_data_with_manual_cache())
    
    # 在 Streamlit 應用中顯示數據
    if data:
        st.write(f"Extracted {len(data)} records:")
        st.table(data)

if __name__ == "__main__":
    main()
方法 3：使用 st.cache_resource
Streamlit 的 @st.cache_resource 可以緩存需要大量計算的資源，但它同樣不支援異步函數。你可以將數據庫連接等重資源操作放入這個裝飾器中，然後在緩存完連接資源後，在同步上下文中進行數據操作。
import streamlit as st
import asyncio
import psycopg

# 緩存資料庫連接（非異步）
@st.cache_resource
def get_connection():
    conn = psycopg.connect(
        "dbname=test user=postgres password=yourpassword host=localhost"
    )
    return conn

# 異步提取數據
async def async_extract_data(conn):
    async_conn = await conn
    async with async_conn.cursor() as cur:
        await cur.execute("SELECT id, name, price FROM products")
        data = await cur.fetchall()
    return data

# 使用 Streamlit
def main():
    st.title("Streamlit Cache with Database Connection")
    
    # 獲取緩存的資料庫連接
    conn = get_connection()
    
    # 提取數據
    data = asyncio.run(async_extract_data(conn))
    
    # 顯示數據
    if data:
        st.write(f"Extracted {len(data)} records:")
        st.table(data)

if __name__ == "__main__":
    main()
總結
    1. asyncio.run(): 當需要將異步函數與 @st.cache_data 一起使用時，可以將異步函數轉換為同步調用。
    2. 手動緩存: 自行實現一個緩存機制，可以控制異步數據的緩存時間和過期邏輯。
    3. @st.cache_resource: 當涉及緩存數據庫連接等重資源時，可以考慮使用此裝飾器緩存資源，然後配合同步邏輯進行操作。
這些方法都可以根據具體需求進行調整，以達到最佳性能和體驗。
