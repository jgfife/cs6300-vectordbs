CS 6300 Vector Database Assignment

## Dataset Setup

1. Download the Wikipedia Movie Plots dataset:
   ```bash
   curl -L -o wikipedia-movie-plots.zip "https://www.kaggle.com/api/v1/datasets/download/jrobischon/wikipedia-movie-plots"
   ```
   
   Note: You may need to set up Kaggle API credentials first. See [Kaggle API documentation](https://www.kaggle.com/docs/api) for setup instructions.

2. Extract the dataset:
   ```bash
   unzip wikipedia-movie-plots.zip
   ```

## Running the ChromaDB Demo

1. Install dependencies:
   ```bash
   make install
   ```

2. Run the ChromaDB program:
   ```bash
   make chroma
   ```

   Or alternatively:
   ```bash
   . .virtual_environment/bin/activate
   python3 src/chroma.py
   ```
