def build_prompt(question, context):
    """
    Create a grounded prompt using only retrieved PDF content.
    """

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the information
provided in the document context below.

Do not use outside knowledge.
Do not make up or assume information.

If the answer cannot be found in the provided context,
say:

"I cannot find the answer in the uploaded documents."

Always provide the source filename and page number
when answering.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    return prompt