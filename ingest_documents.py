import os
import json
import fitz  # PyMuPDF
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util

'''
todo:
- extraction improvement:
    - when a table continues to the next page, include it in the previous table
    - preserve the order of image and table inside the text
'''

class PDFExtractor:
    def __init__(self, output_folder, device=None):
        self.device = "cuda"
        self.torch_dtype = torch.float16
        self.output_folder = output_folder
        print(f'Loading onto {self.device}')
        
        print("Loading Florence-2-large model and processor...")
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     "microsoft/Florence-2-large",
        #     torch_dtype=self.torch_dtype,
        #     trust_remote_code=True,
        #     revision="main"
        # ).to(self.device)

        # self.processor = AutoProcessor.from_pretrained(
        #     "microsoft/Florence-2-large",
        #     trust_remote_code=True,
        #     revision="main"
        # )
        print("Model and processor loaded successfully.")

        print("Loading embedding model...")
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        self.embed_model_max_tokens = 512
        print("Embedding model loaded successfully.")

        print(f"Making output Folder {self.output_folder}")
        os.makedirs(self.output_folder, exist_ok=True)

    def process_pdfs(self, input_paths):
        """
        Accepts a list of file paths or directories.
        For directories, scans for all PDF files inside.
        """
        all_data = {}
        pdf_list = []

        for path in input_paths:
            if os.path.isdir(path):
                print(f"Scanning folder for PDFs: {path}")
                for filename in os.listdir(path):
                    if filename.lower().endswith(".pdf"):
                        full_path = os.path.join(path, filename)
                        pdf_list.append(full_path)
            elif os.path.isfile(path) and path.lower().endswith(".pdf"):
                pdf_list.append(path)
            else:
                print(f"Skipping invalid input: {path}")

        if not pdf_list:
            print("No valid PDF files found.")
            return

        for pdf_path in pdf_list:
            print(f"\nProcessing PDF: {pdf_path}")
            doc_name = os.path.splitext(os.path.basename(pdf_path))[0]
            pdf_data, full_doc_text = self._extract_from_pdf(pdf_path)

            embedding = self._get_document_embedding(full_doc_text)
            pdf_data["embedding"] = embedding.tolist()

            all_data[doc_name] = pdf_data

        combined_output_path = os.path.join(self.output_folder, "combined_output.json")
        with open(combined_output_path, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=4)

        print(f"All PDFs processed. Combined JSON saved to {combined_output_path}")

    def _extract_from_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        pdf_data = {}
        full_text_accumulator = []

        for page_num, page in enumerate(tqdm(doc, desc='Page Extraction')):
            page_text = ""
            image_text = ""
            table_text = ""

            text_blocks = page.get_text("blocks")
            text_content = "\n".join(block[4] for block in text_blocks if block[4].strip())
            page_text = text_content.strip()

            image_objs = []
            for i, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = page.parent.extract_image(xref)
                ext = base_image["ext"]
                image_path = os.path.join(self.output_folder, f'page_{page_num + 1}_img_{i + 1}.{ext}')
                with open(image_path, "wb") as f:
                    f.write(base_image["image"])
                image_objs.append((img[1], image_path))

            image_descriptions = self._describe_images(image_objs)
            for desc in image_descriptions:
                image_text += f"[Image Description]: {desc}\n\n"

            tables = page.find_tables()
            if tables and tables.tables:
                for table in tables.tables:
                    markdown_table = table.to_pandas().to_markdown(index=False)
                    table_text += f"[Extracted Table]\n{markdown_table}\n\n"

            full_page_text = f"{page_text}\n\n{image_text}\n\n{table_text}".strip()
            full_text_accumulator.append(full_page_text)
            pdf_data[f"page_{page_num + 1}"] = {"page_extraction": full_page_text}

        combined_text = "\n\n".join(full_text_accumulator)
        return pdf_data, combined_text

    def _get_document_embedding(self, full_text):
        """
        Computes document embedding. If the text is too long, splits it into chunks and averages the embeddings.
        """
        words = full_text.split()
        max_words = self.embed_model_max_tokens
        if len(words) <= max_words:
            embedding = self.embed_model.encode(full_text)
        else:
            chunks = [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]
            embeddings = [self.embed_model.encode(chunk) for chunk in chunks]
            embedding = torch.tensor(embeddings).mean(dim=0)

        return embedding

    def _describe_images(self, image_objs):
        descriptions = []
        prompt = "<MORE_DETAILED_CAPTION>"

        if not image_objs:
            return descriptions

        for _, image_path in image_objs:
            image = Image.open(image_path)
            parsed_answer = 'testing tool - image was not parsed yet.'
            descriptions.append(parsed_answer)

        return descriptions


# Example usage
extractor = PDFExtractor(output_folder='./tmp_extract_pdf')
pdf_inputs = ['./nap_alb.pdf', './astro_pdfs2/ZASPE A Code to Measure Stellar Atmospheric Parameters and their Covariance from Spectra.pdf']
result = extractor.process_pdfs(pdf_inputs)
