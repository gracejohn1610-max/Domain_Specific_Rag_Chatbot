from pypdf import PdfReader


def extract_text_from_pdf(uploaded_file):
    """
    Extract text from every page of an uploaded PDF.

    Returns:
        A list containing the text, source filename,
        and page number for each non-empty page.
    """

    reader = PdfReader(uploaded_file)

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text()

        if text and text.strip():

            documents.append({
                "text": text.strip(),
                "source": uploaded_file.name,
                "page": page_number
            })

    return documents