import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser # Changed to String Parser (Safer)
from ingestion import KnowledgeBase

load_dotenv()

class TestGenAgent:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")

        self.llm = ChatGroq(
            temperature=0, 
            model_name="llama-3.3-70b-versatile",
            api_key=api_key
        )
        self.kb = KnowledgeBase()

    def generate_tests(self, user_query: str):
        try:
            # 1. Retrieve relevant docs
            retriever = self.kb.get_retriever()
            
            # Modern LangChain uses 'invoke' instead of 'get_relevant_documents'
            docs = retriever.invoke(user_query)
            
            context_text = "\n\n".join([d.page_content for d in docs])

            if not context_text:
                return {"error": "No relevant documentation found. Please upload documents first."}

            # 2. Define Prompt
            prompt = ChatPromptTemplate.from_template("""
            You are an expert QA Automation Lead. Your goal is to generate comprehensive test cases based STRICTLY on the provided context.
            
            RULES:
            1. Use ONLY the provided context. Do not hallucinate features.
            2. Output must be a JSON list of objects.
            3. Each object must have: Test_ID, Feature, Test_Scenario, Expected_Result, Grounded_In (the source doc).
            4. Do NOT output markdown code blocks (like ```json). Just return the raw JSON string.
            
            CONTEXT:
            {context}
            
            USER REQUEST:
            {query}
            
            OUTPUT JSON:
            """)

            # 3. Chain - Use StrOutputParser to capture raw text first (Debugging friendly)
            chain = prompt | self.llm | StrOutputParser()

            raw_response = chain.invoke({
                "context": context_text,
                "query": user_query
            })

            # 4. Post-process: Extract JSON from the text
            # This handles cases where the LLM adds "Here is the json: ..."
            clean_json = raw_response.strip()
            if "```json" in clean_json:
                clean_json = clean_json.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_json:
                clean_json = clean_json.split("```")[1].split("```")[0].strip()

            # 5. Convert to Python Dict
            return json.loads(clean_json)

        except Exception as e:
            # Print the full error to the terminal so you can see it
            print(f"ERROR in generate_tests: {str(e)}")
            return {"error": "Internal Server Error", "details": str(e)}