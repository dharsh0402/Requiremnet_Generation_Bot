import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

class SYS2Requirement(BaseModel):
    sys_id: str = Field(description="Unique System Requirement ID (e.g., SYS_REQ_001)")
    title: str = Field(description="Short title of the requirement")
    description: str = Field(description="Detailed 'Shall' statement")
    category: str = Field(description="Functional, Non-Functional, Performance, Safety, or Interface")
    verification_criteria: str = Field(description="How this requirement will be verified at system level")

class SWE1Requirement(BaseModel):
    sw_id: str = Field(description="Unique Software Requirement ID (e.g., SW_REQ_001)")
    parent_sys_id: str = Field(description="The ID of the parent SYS.2 requirement")
    title: str = Field(description="Short title of the requirement")
    description: str = Field(description="Detailed technical 'Shall' statement")
    category: str = Field(description="Functional, Non-Functional, etc.")
    verification_criteria: str = Field(description="Technical verification criteria (e.g., unit test, static analysis)")

class RequirementList(BaseModel):
    requirements: List[SYS2Requirement]

class SWERequirementList(BaseModel):
    requirements: List[SWE1Requirement]

class AIEngine:
    def __init__(self, rag_vault=None):
        self.rag_vault = rag_vault
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_groq_api_key_here":
            # Fallback for demo if key is missing
            self.llm = None
        else:
            self.llm = ChatGroq(
                temperature=0.2,
                model_name="llama3-70b-8192",
                groq_api_key=api_key
            )

    async def generate_sys2(self, user_input: str, context: Optional[str] = None) -> List[SYS2Requirement]:
        # Search for context if RAG is available
        rag_context = ""
        if self.rag_vault:
            rag_context = self.rag_vault.search_context(user_input)
            
        if not self.llm:
            return self._mock_sys2()

        parser = JsonOutputParser(pydantic_object=RequirementList)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Automotive Systems Engineer specializing in ASPICE SYS.2.
            Your task is to transform stakeholder needs into high-level System Requirements.
            
            Use the following existing requirements as context for consistency:
            {rag_context}
            
            ASPICE SYS.2 Principles:
            1. Uniqueness: Each requirement must have a unique ID.
            2. Testability: Each requirement must have clear verification criteria.
            3. Clarity: Use formal "The system shall..." language.
            4. Categorization: Separate functional and non-functional aspects.
            
            Industry Context: {context}
            
            Output ONLY a JSON object matching the requested schema."""),
            ("human", "Generate system requirements for the following feature: {user_input}")
        ])

        chain = prompt | self.llm | parser
        result = chain.invoke({
            "user_input": user_input, 
            "context": context or "General Automotive",
            "rag_context": rag_context
        })
        return result["requirements"]

    async def generate_swe1(self, sys2_requirement: dict) -> List[SWE1Requirement]:
        if not self.llm:
            return self._mock_swe1(sys2_requirement["sys_id"])

        parser = JsonOutputParser(pydantic_object=SWERequirementList)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Automotive Software Architect specializing in ASPICE SWE.1.
            Your task is to decompose a System Requirement (SYS.2) into detailed Software Requirements (SWE.1).
            
            ASPICE SWE.1 Principles:
            1. Traceability: Link every software requirement to its parent system requirement.
            2. Atomicity: Software requirements should be detailed enough for developers to implement.
            3. Testability: Define how it will be verified (e.g., Unit Test, HIL Test).
            
            Output ONLY a JSON object matching the requested schema."""),
            ("human", "Decompose this system requirement into software requirements: {sys2_data}")
        ])

        chain = prompt | self.llm | parser
        result = chain.invoke({"sys2_data": str(sys2_requirement)})
        return result["requirements"]

    def _mock_sys2(self):
        return [
            SYS2Requirement(
                sys_id="SYS_REQ_001",
                title="Automatic Braking Logic",
                description="The system shall initiate automatic emergency braking when an obstacle is detected within 5 meters at speeds above 20km/h.",
                category="Safety",
                verification_criteria="Verified by System Integration Test SIT_001"
            )
        ]

    def _mock_swe1(self, parent_id):
        return [
            SWE1Requirement(
                sw_id=f"SW_REQ_{parent_id}_01",
                parent_sys_id=parent_id,
                title="Obstacle Distance Calculation",
                description="The software shall calculate the distance to the nearest obstacle using fused data from Radar and Camera sensors with a latency < 50ms.",
                category="Functional",
                verification_criteria="Unit Test UT_DISTANCE_CALC"
            )
        ]
