from fastapi import FastAPI, Request, HTTPException
import uvicorn
from langchain_community.llms import HuggingFaceHub
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

app = FastAPI()

# Initialize the Hugging Face model
llm = HuggingFaceHub(repo_id="facebook/bart-large-cnn", model_kwargs={"temperature": 0.7})

# Load the summarization chain with the LLM
summarization_chain = load_summarize_chain(llm=llm)


@app.post("/summarize")
async def summarize(request: Request):
    data = await request.json()
    text = data.get("text", "")

    # Create a Document object for the text
    document = Document(page_content=text)

    try:
        # Generate summary using LangChain summarization chain
        summary = summarization_chain.run([document])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Debug, I stumbled upon some mistakes in summariztion procces, so I had to add theese
    print(f"Original Text: {text}")
    print(f"Summary: {summary}")

    # Return the summary in a JSON response
    return {"summary": summary}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)