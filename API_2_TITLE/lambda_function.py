from fastapi import FastAPI
from mangum import Mangum
from endpoints import register_routes  # import your routes

# Create FastAPI app
app = FastAPI()

# Register routes from endpoints.py
register_routes(app)

# Lambda handler
lambda_handler = Mangum(app)
