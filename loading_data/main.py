import sys
from datasets import load_dataset

class HFDataIndexer:
    """Lazy indexer for Hugging Face datasets."""

    def __init__(self, path, name=None, split='train'):
        self.path = path
        self.split = split
        try:
            self.dataset = load_dataset(path, name, split=split, streaming=True)
            
            # Peek at one row to get column names without loading the rest
            first_row = next(iter(self.dataset))
            self.columns = list(first_row.keys())
            
            print(f"--- Connected to {path} ({split}) ---")
            print(f"Columns available: {self.columns}")
        except Exception as e:
            print(f"Connection Error: {e}")
            sys.exit(1)


    def get_rows(self, start=0, count=1, target_columns=None):
        """
        Stream a specific range of rows. Returns a generator to 
        minimize memory footprint.
        """
        cols = target_columns if target_columns else self.columns
        
        # "lazy" indexing in streaming mode uses skip() and take()
        subset = self.dataset.skip(start).take(count)
        
        for row in subset:
            yield { col: row.get(col) for col in cols }


    def get_cell(self, row_idx, col_name):
        """
        Grab a single specific 'cell'.
        """
        if col_name not in self.columns:
            return None
            
        # skip to row_idx, take 1 row, get the first item
        row = next(iter(self.dataset.skip(row_idx).take(1)), None)

        return row.get(col_name) if row else None



if __name__ == "__main__":
    indexer = HFDataIndexer("TIGER-Lab/MMLU-Pro", split="validation")

    # 1. Grab a specific cell (Row 5, Column 'question')
    question_5 = indexer.get_cell(5, "question")
    print(f"Cell Index [5, 'question']:\n{question_5[:100]}...\n")

    # 2. Iterate over a range (Rows 10 to 13)
    print("Streaming Rows 10 - 13 ...\n\n")

    fields = ['question', 'options', 'answer']
    data_stream = indexer.get_rows(start=10, count=3, target_columns=fields)

    for i, entry in enumerate(data_stream, start=10):
        extracted_fields = [f"{f}:\n{entry.get(f, None)}\n\n" for f in fields]
        print(f"Row {i}:\n\n" + "\n".join(extracted_fields) + "\n" + "-" * 25 + "\n\n")
        