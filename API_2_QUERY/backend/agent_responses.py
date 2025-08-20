from pydantic import BaseModel, Field, ValidationError, field_validator
from typing import List, Optional, Union
import json
import re


def convert_step_dict_to_list(step_data: Union[List[str], dict]) -> List[str]:
    if isinstance(step_data, dict):
        try:
            return [step_data[k] for k in sorted(
                step_data.keys(),
                key=lambda x: int(''.join(filter(str.isdigit, str(x))) or 0)
            )]
        except Exception:
            return [step_data[k] for k in sorted(step_data.keys(), key=str)]
    return step_data


def clean_json_string(raw: str) -> str:
    """
    Removes markdown code fences and ensures valid JSON.
    """
    if isinstance(raw, str):
        raw = raw.strip()
        # Remove starting ```json or ``` if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        # Remove ending ``` if present
        raw = re.sub(r"\s*```$", "", raw)
    return raw


# ---------------------- Intent 1: Fault Diagnosis ----------------------
class FaultDiagnosisResponse(BaseModel):
    """
    Responds to fault queries with causes, actions, and safety information.
    """
    issue_identified: str = Field(..., description="Summary of the identified issue (non-empty).")
    likely_causes: Union[List[str], dict] = Field(..., description="List of possible root causes (at least 1).")
    recommended_actions: Union[List[str], dict] = Field(..., description="Corrective actions (at least 1).")
    precautionary_notes: Optional[Union[List[str], dict]] = Field(None, description="Safety precautions (optional).")

    # ---------------- Validation ----------------
    @field_validator("issue_identified")
    def issue_not_empty(cls, v: str):
        if not v or not v.strip():
            raise ValueError("issue_identified cannot be empty")
        return v

    @field_validator("likely_causes", "recommended_actions")
    def list_not_empty(cls, v, info):
        if (isinstance(v, list) and not v) or (isinstance(v, dict) and not v):
            raise ValueError(f"{info.field_name} cannot be empty")
        return v

    def post_process(self):
        self.likely_causes = convert_step_dict_to_list(self.likely_causes)
        self.recommended_actions = convert_step_dict_to_list(self.recommended_actions)
        if self.precautionary_notes:
            self.precautionary_notes = convert_step_dict_to_list(self.precautionary_notes)

    @staticmethod
    def get_prompt_schema():
        return {
            "issue_identified": "Error Code 3500 on AGT-101",
            "likely_causes": [
                "Motor overload during the mixing phase",
                "Loose connection in phase 3 of the supply cable"
            ],
            "recommended_actions": [
                "Stop the agitator immediately using the emergency switch.",
                "Inspect all power connections to the motor and VFD.",
                "Reset the overload relay and monitor for re-occurrence.",
                "If the issue persists, replace the thermal sensor as per SOP."
            ],
            "precautionary_notes": [
                "Ensure the agitator is completely isolated before inspection.",
                "Wear arc-protection PPE during electrical checks."
            ]
        }


# ---------------------- Intent 2: Operational Guidance ----------------------
class OperationalGuidanceResponse(BaseModel):
    """
    Guides operator on how to perform a specific procedure or task.
    """
    task: str = Field(..., description="Procedure being performed (non-empty).")
    tools_needed: Union[List[str], dict] = Field(..., description="Tools required (at least 1).")
    step_by_step_procedure: Union[List[str], dict] = Field(..., description="Procedure steps (at least 1).")
    safety_checklist: Optional[Union[List[str], dict]] = Field(None, description="Safety precautions (optional).")

    # ---------------- Validation ----------------
    @field_validator("task")
    def task_not_empty(cls, v: str):
        if not v or not v.strip():
            raise ValueError("task cannot be empty")
        return v

    @field_validator("tools_needed", "step_by_step_procedure")
    def list_not_empty(cls, v, info):
        if (isinstance(v, list) and not v) or (isinstance(v, dict) and not v):
            raise ValueError(f"{info.field_name} cannot be empty")
        return v

    def post_process(self):
        self.tools_needed = convert_step_dict_to_list(self.tools_needed)
        self.step_by_step_procedure = convert_step_dict_to_list(self.step_by_step_procedure)
        if self.safety_checklist:
            self.safety_checklist = convert_step_dict_to_list(self.safety_checklist)

    @staticmethod
    def get_prompt_schema():
        return {
            "task": "Core Recalibration – AGT-101",
            "tools_needed": [
                "Digital Vernier Caliper",
                "Calibration Probe",
                "Allen Key Set"
            ],
            "step_by_step_procedure": [
                "Power down AGT-101 using the main breaker.",
                "Remove the top cover plate with a 5mm Allen key.",
                "Attach the calibration probe to the central shaft.",
                "Use the calibration software (v2.1) to adjust offset and gain.",
                "Tighten fixtures and verify alignment.",
                "Restart the agitator and verify performance."
            ],
            "safety_checklist": [
                "Confirm isolation using lockout-tagout",
                "Verify 0V before manual adjustments"
            ]
        }


# ---------------------- Common: No Result ----------------------
class NoResultResponse(BaseModel):
    """
    Represents the 'no relevant document found' case.
    """
    message: str = Field(..., pattern=r"^No relevant document found$", description="Fixed message string.")

    @staticmethod
    def get_prompt_schema():
        return {"message": "No relevant document found"}


# ---------------------- Validator Function ----------------------
def validate_llm_response(response_data: Union[dict, str], intent_type: str):
    model_map = {
        "fault_diagnosis": FaultDiagnosisResponse,
        "operational_guidance": OperationalGuidanceResponse,
    }

    try:
        # Convert string to JSON if needed
        if isinstance(response_data, str):
            response_data = clean_json_string(response_data)
            response_data = json.loads(response_data)

        # Case 1: Explicit "No relevant document found"
        if isinstance(response_data, dict) and "message" in response_data:
            return NoResultResponse(**response_data).model_dump()

        # Case 2: Normal intent-based parsing
        model = model_map.get(intent_type.lower())
        if not model:
            raise ValueError(f"Unknown intent type: {intent_type}")

        parsed = model(**response_data)

        # Post-processing (dict → list conversion)
        if hasattr(parsed, "post_process"):
            parsed.post_process()

        return parsed.model_dump()

    except (ValidationError, json.JSONDecodeError) as e:
        raise ValueError(f"❌ LLM response failed validation:\n{e}")

