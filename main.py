!pip install openai-whisper
!pip install python-dotenv
!pip install --upgrade transformers
!pip install -U langchain-community
!pip install pinecone-client
!pip install -U langchain-openai
!pip install -U langchain-pinecone
!pip install --upgrade scikit-learn


import torch
from torch import cuda

# Set device to GPU if available; otherwise, use CPU
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables from "environment.env" file
load_dotenv(find_dotenv("environment.env"))

import yt_dlp
import logging

def download_audio_from_youtube(playlist_url, output_path="/notebooks/Files/%(title)s.%(ext)s"):
    """
    Downloads audio from a YouTube playlist or video and saves it as an mp3 file.

    Args:
        playlist_url (str): URL of the YouTube playlist or video to download.
        output_path (str, optional): Path format for saving the audio file. 
                                     Defaults to '/notebooks/Files/%(title)s.%(ext)s'.

    Returns:
        None
    """
    # Configure yt-dlp options for audio download
    ydl_opts = {
        'format': 'bestaudio/best',  # Best available audio quality
        'outtmpl': output_path,      # Output file naming format
        'postprocessors': [{         # Convert audio to mp3
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'ignoreerrors': True,        # Continue to next video on error
    }

    # Initialize logging for better error tracking
    logging.basicConfig(level=logging.INFO)
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([playlist_url])
        logging.info("Download completed successfully.")
    except Exception as e:
        logging.error(f"Error downloading audio: {e}")

import whisper
import os
import logging
from typing import Optional

def transcribe_audio_files(audio_dir: str, transcription_dir: str, model_type: str = "medium") -> Optional[None]:
    """
    Transcribes audio files in the specified directory using the Whisper model.

    Args:
        audio_dir (str): Directory containing audio files to transcribe.
        transcription_dir (str): Directory to save transcription files.
        model_type (str, optional): Type of Whisper model to use. Defaults to "medium".

    Returns:
        None
    """
    # Initialize logging
    logging.basicConfig(level=logging.INFO)

    # Load Whisper model
    try:
        model = whisper.load_model(model_type)
        logging.info(f"Loaded Whisper model: {model_type}")
    except Exception as e:
        logging.error(f"Error loading model '{model_type}': {e}")
        return

    # Check if audio directory exists
    if not os.path.isdir(audio_dir):
        logging.error(f"Audio directory '{audio_dir}' does not exist.")
        return

    # Ensure transcription directory exists
    os.makedirs(transcription_dir, exist_ok=True)
    
    # Define supported audio file extensions
    supported_extensions = (".mp3", ".wav", ".m4a")
    
    # Get list of supported audio files in the audio directory
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith(supported_extensions)]
    if not audio_files:
        logging.info("No audio files found in the specified directory.")
        return
    
    # Transcribe each audio file and save it as a .txt file
    for audio_file in audio_files:
        audio_path = os.path.join(audio_dir, audio_file)
        
        try:
            # Transcribe audio file
            transcription = model.transcribe(audio_path)
            
            # Save transcription to a .txt file
            transcription_file = os.path.join(
                transcription_dir, os.path.splitext(audio_file)[0] + ".txt"
            )
            with open(transcription_file, "w") as f:
                f.write(transcription['text'])
            
            logging.info(f"Transcribed and saved: {transcription_file}")
        
        except Exception as e:
            logging.error(f"Error transcribing file '{audio_file}': {e}")

import os
import logging
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define the path to the transcription folder
transcription_folder = "/notebooks/Files/TranscriptionMedium/"
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=300)

# Retrieve the list of files in the transcription folder
transcription_files = os.listdir(transcription_folder)
if transcription_files:
    vectorized_segments = []
    global_counter = 0  # Initialize global counter

    for transcription_file in transcription_files:
        transcription_path = os.path.join(transcription_folder, transcription_file)
        if os.path.isfile(transcription_path):
            logging.info(f"Processing file: {transcription_path}")

            # Read file content
            with open(transcription_path, 'r', encoding='utf-8') as file:
                transcription_text = file.read()

            # Split text into segments
            segments = text_splitter.split_text(transcription_text)

            # Vectorize segments and store them with metadata
            if segments:
                embeddings = embedding_model.embed_documents(segments)

                for segment, vector in zip(segments, embeddings):
                    vectorized_segments.append({
                        "id": f"segment_{global_counter}",
                        "text": segment,
                        "embedding": vector
                    })
                    global_counter += 1
            else:
                logging.warning(f"No segments to vectorize in file {transcription_file}.")
        else:
            logging.warning(f"Skipping {transcription_file}, not a file.")

    logging.info("Vectorization complete.")
else:
    logging.warning("No files found in the transcription folder.")

# Print 5 segments to check structure
for segment in vectorized_segments[700:705]:
    print("Segment ID:", segment["id"])
    print("Text:", segment["text"][:100] + "...")  # Show first 100 characters of text
    print("Embedding Length:", len(segment["embedding"]))
    print("Embedding Sample:", segment["embedding"][:5])  # Show first 5 values in the embedding
    print("-" * 50)

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Tuple

# Define a function to search for similar segments
def search_similar_segment(query_text: str, embedding_model, vectorized_segments: List[dict], top_k: int = 1) -> List[Tuple[str, float]]:
    """
    Search for segments similar to the query text based on cosine similarity.

    Args:
        query_text (str): The text to search for similarities.
        embedding_model: The embedding model used to generate embeddings.
        vectorized_segments (List[dict]): List of segments with embeddings.
        top_k (int): Number of top similar segments to return.

    Returns:
        List[Tuple[str, float]]: List of tuples containing text of similar segments and their similarity scores.
    """
    if not vectorized_segments:
        print("No vectorized segments available.")
        return []

    # Generate embedding for the query text
    query_embedding = embedding_model.embed_documents([query_text])[0]

    # Extract embeddings from vectorized_segments
    segment_embeddings = [segment["embedding"] for segment in vectorized_segments]

    # Calculate cosine similarity with each stored embedding
    similarities = cosine_similarity([query_embedding], segment_embeddings).flatten()

    # Find the top_k most similar segments
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]
    similar_segments = [(vectorized_segments[i]["text"], similarities[i]) for i in top_k_indices]

    return similar_segments

# Example usage
query_text = "negatives for syntropic agriculture"
similar_segments = search_similar_segment(query_text, embedding_model, vectorized_segments)

# Print the results
for segment, similarity in similar_segments:
    print(f"Similarity: {similarity:.2f} - Segment: {segment[:100]}...")  # Truncate for display


# Display the total number of vectorized segments
print(f"Total segments vectorized: {len(vectorized_segments)}")

# Check for consistency in embedding dimensions
embedding_lengths = {len(segment["embedding"]) for segment in vectorized_segments}
print("Unique embedding lengths:", embedding_lengths)

# Confirm consistency
if len(embedding_lengths) == 1:
    print("All embeddings have consistent dimensions.")
else:
    print("Warning: Inconsistent embedding dimensions found.")

# Check if each segment has text and a non-empty embedding
valid_segments = all(segment["text"] and segment["embedding"] for segment in vectorized_segments)

if valid_segments:
    print("All segments are valid.")
else:
    print("Warning: Some segments are missing text or embeddings.")

from pinecone import ServerlessSpec
import os

# Securely fetch Pinecone API key and ensure it is set
api_key = os.getenv("PINECONE_API_KEY")
if api_key:
    os.environ["PINECONE_API_KEY"] = api_key
else:
    raise ValueError("PINECONE_API_KEY not found. Please set it in your environment variables.")

# Define the serverless specification for Pinecone
spec = ServerlessSpec(
    cloud="aws", 
    region="us-east-1"
)

import os
import time
from pinecone import Pinecone


# Initialize Pinecone client with the updated API
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

index_name = "agricultura-sintropica"

# Check if the index exists; create if it doesn’t
if index_name not in [index_info["name"] for index_info in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,  # OpenAI embedding dimension
        metric="dotproduct",
        spec=spec
    )
    # Wait for the index to be ready
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

# Connect to the index for upsertion
index = pc.Index(index_name)
time.sleep(1)
# view index stats
index.describe_index_stats()

from tqdm.auto import tqdm

# Define batch size for upsertion
batch_size = 50

# Upsert embeddings to Pinecone in batches
for i in tqdm(range(0, len(vectorized_segments), batch_size), desc="Upserting Batches"):
    batch = vectorized_segments[i:i + batch_size]
    ids = [segment["id"] for segment in batch]
    embeddings = [segment["embedding"] for segment in batch]
    metadatas = [{"text": segment["text"]} for segment in batch]

    # Attempt upsertion of each batch to Pinecone
    try:
        index.upsert(vectors=list(zip(ids, embeddings, metadatas)))
    except Exception as e:
        print(f"Error upserting batch {i // batch_size + 1}: {e}")

print("All segments have been upserted into Pinecone.")

# View and display index stats
index_stats = index.describe_index_stats()
print("Index statistics:", index_stats)

import os

# Set environment variables for LangChain tracing and endpoint configuration
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "pr-drab-platinum-76"

# Verify environment variables are set
required_vars = ["LANGCHAIN_TRACING_V2", "LANGCHAIN_ENDPOINT", "LANGCHAIN_PROJECT"]
for var in required_vars:
    if var in os.environ:
        print(f"{var} set to: {os.environ[var]}")
    else:
        print(f"Warning: {var} is not set.")

from langchain_openai import OpenAIEmbeddings
import os

# Define the model name
model_name = 'text-embedding-ada-002'

# Retrieve the OpenAI API key securely
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key is not set. Please set OPENAI_API_KEY in environment variables.")

# Initialize the OpenAIEmbeddings model
embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=api_key
)

from langchain_pinecone import PineconeVectorStore

# Initialize PineconeVectorStore with the specified index, embedding model, and text key
try:
    vector_store = PineconeVectorStore(index=index, embedding=embedding_model, text_key="text")
    print("PineconeVectorStore initialized successfully.")
except Exception as e:
    print(f"Error initializing PineconeVectorStore: {e}")


