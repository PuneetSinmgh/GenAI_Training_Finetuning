def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
import wget

filename = 'companyPolicies.txt'
with open(filename, 'r') as file:
    # Read the contents of the file
    contents = file.read()
    print(contents)

#Load and split the documents
loader = TextLoader(filename)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
print(f"Number of documents: {len(texts)}")
print(texts[0])

#Create embeddings
embeddings = HuggingFaceEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)
print('document ingested')
print(f"Number of documents in the vector store: {docsearch._collection.count()}")

#Create retrieval-based QA chain
#LLM model construction¶
model_id = 'ibm/granite-3-2-8b-instruct'

parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,  
    GenParams.MIN_NEW_TOKENS: 130, # this controls the minimum number of tokens in the generated output
    GenParams.MAX_NEW_TOKENS: 256,  # this controls the maximum number of tokens in the generated output
    GenParams.TEMPERATURE: 0.5 # this randomness or creativity of the model's responses
}

credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "api_key": "1rpZ1c6qK0qpSYYfowRwHi3UVC7MDL_Jn_H9yBiBwkQa"
    # uncomment above when running locally
}
project_id = "4e8cb5ed-c2e8-46f0-8f6c-ce1b63ecd70f"

model = Model(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=project_id
)

flan_ul2_llm = WatsonxLLM(model=model)
qa_chain = RetrievalQA.from_chain_type(
    llm=flan_ul2_llm,
    chain_type="stuff",
     retriever=docsearch.as_retriever(),
    return_source_documents=False
)



#Query the chain
query = "What is the company's policy on remote work?"
response = qa_chain.invoke(query)
print("Query:", query)
print("Response:", response)
num_valid = len(train_dataset) - num_train
print(num_valid)

query = "what is mobile policy?"
qa_chain.invoke(query)

#Query the chain
query = "Can you summarize the document for me?"
print("Response:", qa_chain.invoke(query))


def chain_with_llama() :
    model_id = 'meta-llama/llama-3-2-11b-vision-instruct'

    parameters = {
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY,  
        GenParams.MAX_NEW_TOKENS: 256,  # this controls the maximum number of tokens in the generated output
        GenParams.TEMPERATURE: 0.5 # this randomness or creativity of the model's responses
    }

    credentials = {
        "url": "https://us-south.ml.cloud.ibm.com"
    }

    project_id = "skills-network"

    model = Model(
        model_id=model_id,
        params=parameters,
        credentials=credentials,
        project_id=project_id
    )

    llama_3_llm = WatsonxLLM(model=model)
    return RetrievalQA.from_chain_type(
        llm=llama_3_llm,
        chain_type="stuff",
         retriever=docsearch.as_retriever(),
        return_source_documents=False
    )

llama_qa_chain = chain_with_llama()
query = "Can you summarize the document for me?"
llama_qa_chain.invoke(query)