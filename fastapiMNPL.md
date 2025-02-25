fastapi 更多進階範例 tornado async 
要在 FastAPI 中實現進階的非同步功能並通過 Tornado 提供更高的性能，可以採用多種進階方法，比如使用協程、處理複雜的非同步請求、並行請求、以及進行流式數據傳輸。這裡展示一些進階技巧和最佳實踐，以幫助你創建高效的 FHIR Server。
進階 FastAPI 非同步操作
1. 使用協程處理非同步 I/O 操作
FastAPI 使用非同步編程來處理 I/O 密集型操作，例如數據庫查詢、網絡請求等。可以使用 async 和 await 關鍵字來管理協程，避免阻塞主線程。
例如，假設我們有一個需要調用外部 API 的情況，可以用非同步的方式調用，並且進行超時控制：
import httpx
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/async-data")
async def fetch_data():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("https://example.com/api/data")
            response.raise_for_status()  # 若發生錯誤則拋出異常
            return response.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail="External API error")
2. 使用 Background Tasks 進行背景處理
FastAPI 的背景任務功能允許在回應發送後繼續執行特定任務，適用於非即時性操作（例如日誌記錄、郵件通知等）。
from fastapi import BackgroundTasks

async def write_log(message: str):
    with open("log.txt", "a") as log_file:
        log_file.write(f"{message}\n")

@app.post("/process")
async def process_data(background_tasks: BackgroundTasks, data: dict):
    # 處理數據的邏輯
    background_tasks.add_task(write_log, "Data processed successfully")
    return {"status": "Data is being processed"}
3. 流式回應 (Streaming Responses)
流式回應適用於大數據集或長時間運行的過程中逐步回應數據。這裡以查詢和回應大型數據集為例：
from fastapi.responses import StreamingResponse
from typing import List

async def generate_data():
    # 模擬大型數據的分塊生成
    for i in range(10):
        yield f"Chunk {i}\n"
        await asyncio.sleep(1)

@app.get("/stream-data")
async def stream_data():
    return StreamingResponse(generate_data(), media_type="text/plain")
4. 多任務並行處理 (Concurrency)
可以使用 asyncio.gather 在非同步請求中執行多個 I/O 操作，這在需要並行調用多個資源（如多個數據庫或外部 API）時非常有用。
import asyncio
from fastapi import FastAPI

app = FastAPI()

async def fetch_data_from_source_1():
    await asyncio.sleep(1)
    return {"source_1_data": "data from source 1"}

async def fetch_data_from_source_2():
    await asyncio.sleep(2)
    return {"source_2_data": "data from source 2"}

@app.get("/concurrent-fetch")
async def concurrent_fetch():
    data1, data2 = await asyncio.gather(
        fetch_data_from_source_1(),
        fetch_data_from_source_2()
    )
    return {**data1, **data2}
5. 非同步資料庫訪問
為了在 FastAPI 中進行高效的資料庫操作，建議使用非同步 ORM（如 SQLAlchemy 的非同步支持或 Tortoise ORM）來進行資料庫的訪問。
這裡以 Tortoise ORM 為例：
from tortoise.models import Model
from tortoise import fields, Tortoise

class Patient(Model):
    id = fields.IntField(pk=True)
    name = fields.CharField(max_length=50)
    gender = fields.CharField(max_length=10, null=True)
    birth_date = fields.DateField(null=True)

@app.on_event("startup")
async def init_db():
    await Tortoise.init(
        db_url="sqlite://:memory:",
        modules={"models": ["__main__"]}
    )
    await Tortoise.generate_schemas()

@app.get("/patients")
async def get_patients():
    patients = await Patient.all()
    return [{"id": patient.id, "name": patient.name, "gender": patient.gender} for patient in patients]
使用 Tornado 部署 FastAPI 非同步應用
為了將 FastAPI 應用通過 Tornado 部署並發揮非同步特性，我們可以將 FastAPI 應用嵌入到 Tornado 的 ASGI 服務器中：
import tornado.web
import tornado.ioloop
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from tornado.httpserver import HTTPServer

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello from FastAPI running on Tornado!"}

# 使用 Tornado 的 WSGI Handler 將 FastAPI 應用嵌入
class FastAPIHandler(tornado.web.RequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._asgi_handler = WSGIMiddleware(app)

    async def get(self):
        await self._asgi_handler(self.request)

if __name__ == "__main__":
    app = tornado.web.Application([
        (r"/.*", FastAPIHandler),
    ])
    server = HTTPServer(app)
    server.bind(8000)
    server.start()
    tornado.ioloop.IOLoop.current().start()
測試應用程序
這樣的配置會讓 FastAPI 在 Tornado 上運行，並在 8000 端口上提供非同步處理支持。可以使用以下命令啟動應用：
python tornado_server.py
然後訪問 http://localhost:8000 可以看到非同步的 FastAPI 應用已經在 Tornado 上成功運行
使用streamlit fastapi tornado 開發IG 規範符合smart on fhir, FHIR specification ballot2 的 fhir server
構建一個符合 FHIR Implementation Guide (IG)、SMART on FHIR 和 FHIR Specification Ballot 2 規範的 FHIR Server，可以結合 FastAPI、Tornado 和 Streamlit 來完成。這裡的目標是：
    1. FastAPI：用於構建核心 FHIR API，處理 SMART on FHIR 規範的認證和授權。
    2. Tornado：為 FastAPI 提供非同步和高效能的服務器運行支持。
    3. Streamlit：建立用於展示和測試 FHIR 資源的簡單前端。
以下是詳細的分步流程，逐步實現一個符合上述規範的 FHIR 服務器。
步驟 1：建立 FastAPI 應用程序以實現 FHIR 和 SMART on FHIR 規範
首先，設計一個基本的 FHIR Server，支持 Patient 資源的 CRUD 操作和符合 SMART on FHIR 的 OAuth 2.0 認證。
安裝依賴
pip install fastapi pydantic[dotenv] uvicorn tornado streamlit authlib
1.1 配置 OAuth 2.0 認證
SMART on FHIR 使用 OAuth 2.0 來進行授權。我們將使用 Authlib 庫來實現一個基於 OAuth 的認證服務。
# auth.py
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2AuthorizationCodeBearer
from authlib.integrations.starlette_client import OAuth
from starlette.middleware.sessions import SessionMiddleware
import os

app = FastAPI()

# 設置 OAuth2 的配置（可以通過環境變量或配置文件）
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SESSION_SECRET", "mysecret"))

oauth = OAuth(app)
oauth.register(
    name='fhir',
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    authorize_url="https://example.com/oauth2/authorize",
    access_token_url="https://example.com/oauth2/token",
    client_kwargs={"scope": "openid profile"}
)

oauth2_scheme = OAuth2AuthorizationCodeBearer(authorizationUrl="https://example.com/oauth2/authorize")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    user = await oauth.fhir.parse_id_token(token)
    if not user:
        raise HTTPException(status_code=403, detail="Invalid authentication")
    return user
1.2 定義 FHIR 資源模型
根據 FHIR 標準定義 Patient 資源的模型：
# models.py
from pydantic import BaseModel, Field
from typing import Optional
import uuid

class Patient(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    resourceType: str = "Patient"
    name: str
    gender: Optional[str]
    birthDate: Optional[str]
1.3 創建 CRUD API
實現 FHIR 的 CRUD 操作，並確保需要授權的端點只允許經過授權的用戶訪問。
# main.py
from fastapi import FastAPI, Depends, HTTPException, status
from models import Patient
from auth import get_current_user

app = FastAPI()

# 模擬的數據庫
patients_db = {}

@app.post("/Patient", response_model=Patient, status_code=status.HTTP_201_CREATED)
async def create_patient(patient: Patient, user: dict = Depends(get_current_user)):
    patients_db[patient.id] = patient
    return patient

@app.get("/Patient/{patient_id}", response_model=Patient)
async def get_patient(patient_id: str, user: dict = Depends(get_current_user)):
    patient = patients_db.get(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient

@app.put("/Patient/{patient_id}", response_model=Patient)
async def update_patient(patient_id: str, updated_patient: Patient, user: dict = Depends(get_current_user)):
    if patient_id not in patients_db:
        raise HTTPException(status_code=404, detail="Patient not found")
    patients_db[patient_id] = updated_patient
    return updated_patient

@app.delete("/Patient/{patient_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_patient(patient_id: str, user: dict = Depends(get_current_user)):
    if patient_id in patients_db:
        del patients_db[patient_id]
    else:
        raise HTTPException(status_code=404, detail="Patient not found")
步驟 2：使用 Tornado 部署 FastAPI 應用
Tornado 可以作為一個高效能的 ASGI 服務器，將 FastAPI 應用部署在 Tornado 上：
# tornado_server.py
import tornado.web
import tornado.ioloop
from fastapi import FastAPI
from auth import app as fastapi_app
from fastapi.middleware.wsgi import WSGIMiddleware

class FastAPIHandler(tornado.web.RequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._asgi_handler = WSGIMiddleware(fastapi_app)

    async def get(self):
        await self._asgi_handler(self.request)

if __name__ == "__main__":
    app = tornado.web.Application([
        (r"/.*", FastAPIHandler),
    ])
    app.listen(8000)
    tornado.ioloop.IOLoop.current().start()
步驟 3：使用 Streamlit 開發前端應用
Streamlit 可以用來創建一個用於展示 FHIR 資源的前端。這裡展示如何訪問並展示 Patient 資源：
# streamlit_app.py
import streamlit as st
import requests

API_URL = "http://localhost:8000"
token = st.text_input("Enter your token:", type="password")

st.title("FHIR Patient Resource Viewer")

headers = {"Authorization": f"Bearer {token}"}

if st.button("Fetch Patients"):
    response = requests.get(f"{API_URL}/Patient", headers=headers)
    if response.status_code == 200:
        patients = response.json()
        for patient in patients:
            st.write(f"ID: {patient['id']}")
            st.write(f"Name: {patient['name']}")
            st.write(f"Gender: {patient.get('gender', 'N/A')}")
            st.write(f"Birth Date: {patient.get('birthDate', 'N/A')}")
            st.write("---")
    else:
        st.error("Failed to fetch patients.")
測試與運行
    1. 啟動 Tornado 伺服器來運行 FastAPI 應用：
       python tornado_server.py
    2. 啟動 Streamlit 應用：
       streamlit run streamlit_app.py
    3. 使用瀏覽器訪問 http://localhost:8501 來查看 Streamlit 前端，並通過輸入 token 來獲取患者資料。
這樣的一個架構包含了符合 SMART on FHIR 和 FHIR 規範的認證和資源訪問設計，並通過 FastAPI、Tornado 和 Streamlit 的結合構建了一個完整的 FHIR Server。
