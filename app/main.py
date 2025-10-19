from typing import Dict, Any

from fastapi import FastAPI, Header
from pydantic import BaseModel

from app.agents.langgraph_react_agent import run_agent


app = FastAPI(title="LangGraph ReAct Agent", version="0.1.0")


class MessageRequest(BaseModel):
    message: str
    session_id: str | None = None


class LGRequest(BaseModel):
    message: str
    session_id: str | None = None
    system_prompt: str | None = None
    provider: str = "openai"
    model_name: str = "gpt-4o"


@app.post("/v1/langgraph/respond")
def langgraph_respond(body: LGRequest, x_user_id: str = Header(...)) -> Dict[str, Any]:
    result = run_agent(
        x_user_id, 
        body.session_id, 
        body.message, 
        body.system_prompt,
        provider=body.provider,
        model_name=body.model_name
    )
    return {
        "user_id": x_user_id,
        "session_id": body.session_id,
        "provider": body.provider,
        "model_name": body.model_name,
        "result": result,
    }



