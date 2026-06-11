import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import (  PyPDFLoader )
from langchain_text_splitters import (    RecursiveCharacterTextSplitter )
from langchain_chroma import (    Chroma  )
from langchain_openai import (  OpenAIEmbeddings,    ChatOpenAI )
from langchain_classic.chains import (   create_retrieval_chain )
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import (   ChatPromptTemplate )

st.title("📄 PDF File Reader")
st.write("----------------")


openai_key = st.text_input(  "OPENAI_API_KEY",    type="password" )

uploaded_file = st.file_uploader(   "PDF 파일을 올려주세요",   type=["pdf"] )
st.write("----------------")

def pdf_to_document(uploaded_file):
    """    Streamlit 업로드 PDF를
    LangChain Document 형태로 변환
    """
    # 임시 폴더 생성
    temp_dir = tempfile.TemporaryDirectory()

    # 임시 PDF 파일
    temp_filepath = os.path.join(     temp_dir.name,    uploaded_file.name    )

    with open(   temp_filepath,    "wb"  ) as f:
        f.write(    uploaded_file.getvalue()    )

    loader = PyPDFLoader(   temp_filepath   )

    pages = loader.load()
    return pages

if uploaded_file is not None:
    pages = pdf_to_document(   uploaded_file   )
    st.success(   f"PDF 페이지 : {len(pages)}"  )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    texts = text_splitter.split_documents(    pages   )

    st.info(  f"문서 조각 : {len(texts)}"  )

    embeddings = OpenAIEmbeddings(  api_key=openai_key   )

    db = Chroma.from_documents(
        documents=texts,
        embedding=embeddings
    )

    retriever = db.as_retriever(
        search_kwargs={
            "k":3
        }
    )

    st.header(   "PDF에게 질문하세요"   )
    question = st.text_input(   "질문 입력"    )

    if st.button(   "질문하기"   ):
        if not question.strip():
            st.warning( "질문을 입력하세요"   )
        else:
            with st.spinner(  "답변 생성중..."     ):
                llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0,
                    api_key=openai_key,
                    streaming=True,
                )

                prompt = ChatPromptTemplate.from_template(
                    """
                    당신은 PDF 분석 AI 입니다.
                    Context:   {context}
                    Question:  {input}
                    답변:
                    """
                )

                document_chain = create_stuff_documents_chain(llm, prompt)
                qa_chain = create_retrieval_chain(retriever, document_chain)

                def stream_answer():
                    for chunk in qa_chain.stream({"input": question}):
                        if answer_chunk := chunk.get("answer"):
                            yield answer_chunk

                st.write_stream(stream_answer)
