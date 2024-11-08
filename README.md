# Syntropic Bot :deciduous_tree:🤖:deciduous_tree:

The Syntropic Bot is a conversational AI assistant designed to answer questions about Syntropic Agroforestry. It leverages AI models to provide informative and context-aware responses, using a combination of speech processing, natural language understanding, and vector similarity search.

# Features

<strong>YouTube Audio Extraction:</strong> Downloads audio content from YouTube videos or playlists related to Syntropic Agroforestry.

<strong>Speech-to-Text Transcription:</strong> Transcribes the downloaded audio using OpenAI's Whisper model.

<strong>Text Embedding and Vectorization:</strong> Processes and vectorizes the transcribed text using OpenAI's text-embedding-ada-002 model.

<strong>Vector Storage with Pinecone:</strong> Stores the embeddings in a Pinecone vector database for efficient retrieval.

<strong>Conversational Retrieval:</strong> Implements a conversational AI chain using LangChain, enabling the bot to answer questions based on the stored knowledge.

<strong>Interactive Chat Interface:</strong> Provides a command-line interface for users to interact with the bot.

# How It Works

<strong>Audio Downloading:</strong> The bot starts by downloading audio from specified YouTube videos or playlists.

<strong>Transcription:</strong> Uses the Whisper model to transcribe the audio files into text.

<strong>Text Processing:</strong> Splits the transcribed text into manageable chunks and generates embeddings for each segment.

<strong>Vector Storage:</strong> Stores the text segments and their corresponding embeddings in a Pinecone vector database.

<strong>Conversational AI:</strong> Sets up a conversational retrieval chain using LangChain and OpenAI's GPT model to answer user queries.

<strong>User Interaction:</strong> Users can ask questions, and the bot will retrieve relevant information from the stored data to provide answers.
