"""
Entity linking functionality using WAT (Web API for Text) service.
"""

import json
import requests
import contextlib
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import Dict, List, Optional, Any
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm progress bar.
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


class WATAnnotation:
    """An entity annotation returned by WAT service."""

    def __init__(self, annotation_data: Dict[str, Any]):
        """
        Initialize annotation from WAT response data.

        Args:
            annotation_data: Dictionary containing annotation information
        """
        # Character offsets
        self.start = annotation_data['start']
        self.end = annotation_data['end']

        # Annotation confidence and probability
        self.rho = annotation_data['rho']
        self.prior_prob = annotation_data['explanation']['prior_explanation']['entity_mention_probability']

        # Annotated text span
        self.spot = annotation_data['spot']

        # Wikipedia entity information
        self.wikipedia_id = annotation_data['id']
        self.wikipedia_title = annotation_data['title']

    def json_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation of the annotation."""
        return {
            'wikipedia_title': self.wikipedia_title,
            'wikipedia_id': self.wikipedia_id,
            'start': self.start,
            'end': self.end,
            'rho': self.rho,
            'prior_prob': self.prior_prob,
            'spot': self.spot
        }


class WATEntityLinker:
    """
    Entity linker using the WAT (Web API for Text) service.
    """

    def __init__(self, gcube_token: str, timeout: int = 30, max_retries: int = 3):
        """
        Initialize the WAT entity linker.

        Args:
            gcube_token: GCUBE token for WAT API access
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.gcube_token = gcube_token
        self.timeout = timeout
        self.max_retries = max_retries
        self.wat_url = 'https://wat.d4science.org/wat/tag/tag'

        # WAT API parameters
        self.default_params = [
            ("gcube-token", self.gcube_token),
            ("lang", 'en'),
            ("tokenizer", "nlp4j"),
            ('debug', 9),
            ("method",
             "spotter:includeUserHint=true:includeNamedEntity=true:includeNounPhrase=true,"
             "prior:k=50,filter-valid,centroid:rescore=true,topk:k=5,voting:relatedness=lm,"
             "ranker:models=0046.models,confidence:models=pruner-wiki.linear")
        ]

    def link_entities(self, doc_id: str, text: str) -> str:
        """
        Perform entity linking on a single document.

        Args:
            doc_id: Document identifier
            text: Document text to process

        Returns:
            JSON string containing entity linking results
        """
        for attempt in range(self.max_retries):
            try:
                # Prepare request parameters
                params = self.default_params + [("text", text)]

                # Make request to WAT API
                response = requests.get(
                    self.wat_url,
                    params=params,
                    timeout=self.timeout
                )
                response.raise_for_status()

                # Parse response
                response_data = response.json()
                annotations = [WATAnnotation(a) for a in response_data.get('annotations', [])]
                json_annotations = [ann.json_dict() for ann in annotations]

                return json.dumps({
                    'doc_id': doc_id,
                    'entities': json_annotations
                })

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout for doc {doc_id}, attempt {attempt + 1}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    logger.error(f"Max retries exceeded for doc {doc_id}")
                    return json.dumps({'doc_id': doc_id, 'entities': []})

            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error for doc {doc_id}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    return json.dumps({'doc_id': doc_id, 'entities': []})

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"JSON parsing error for doc {doc_id}: {e}")
                return json.dumps({'doc_id': doc_id, 'entities': []})

            except Exception as e:
                logger.error(f"Unexpected error for doc {doc_id}: {e}")
                return json.dumps({'doc_id': doc_id, 'entities': []})

        return json.dumps({'doc_id': doc_id, 'entities': []})

    def link_corpus(self, docs: Dict[str, str], num_workers: int = 4) -> List[str]:
        """
        Perform entity linking on a corpus of documents.

        Args:
            docs: Dictionary mapping doc_id -> document_text
            num_workers: Number of parallel workers

        Returns:
            List of JSON strings containing entity linking results
        """
        logger.info(f"Starting entity linking for {len(docs)} documents with {num_workers} workers")

        with tqdm_joblib(tqdm(desc="Entity linking progress", total=len(docs))) as progress_bar:
            results = Parallel(n_jobs=num_workers, backend='multiprocessing')(
                delayed(self.link_entities)(doc_id, text)
                for doc_id, text in docs.items()
            )

        logger.info("Entity linking completed")
        return results

    def process_jsonl_file(self,
                           input_file: str,
                           output_file: str,
                           text_field: str = 'text',
                           id_field: str = 'doc_id',
                           num_workers: int = 4,
                           max_docs: Optional[int] = None) -> None:
        """
        Process a JSONL file and add entity annotations.

        Args:
            input_file: Path to input JSONL file
            output_file: Path to output JSONL file
            text_field: Field name containing document text
            id_field: Field name containing document ID
            num_workers: Number of parallel workers
            max_docs: Maximum number of documents to process (None for all)
        """
        # Load documents
        docs = {}
        doc_count = 0

        logger.info(f"Loading documents from {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, desc="Loading documents"), 1):
                try:
                    doc = json.loads(line.strip())

                    if text_field not in doc or id_field not in doc:
                        logger.warning(f"Missing required fields on line {line_num}")
                        continue

                    doc_id = doc[id_field]
                    text = doc[text_field]

                    if not text or not doc_id:
                        continue

                    docs[doc_id] = text
                    doc_count += 1

                    if max_docs and doc_count >= max_docs:
                        break

                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num}: {e}")
                    continue

        logger.info(f"Loaded {len(docs)} documents for entity linking")

        # Perform entity linking
        results = self.link_corpus(docs, num_workers)

        # Write results
        logger.info(f"Writing results to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(result + '\n')

        logger.info(f"Entity linking completed. Results saved to {output_file}")


def load_documents_for_linking(doc_file: str,
                               text_field: str = 'text',
                               id_field: str = 'doc_id',
                               max_docs: Optional[int] = None) -> Dict[str, str]:
    """
    Load documents from file for entity linking.

    Args:
        doc_file: Path to document file (JSONL format)
        text_field: Field name containing document text
        id_field: Field name containing document ID
        max_docs: Maximum number of documents to load

    Returns:
        Dictionary mapping doc_id -> document_text
    """
    docs = {}

    with open(doc_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading documents"):
            if max_docs and len(docs) >= max_docs:
                break

            try:
                doc = json.loads(line.strip())

                if text_field in doc and id_field in doc:
                    doc_id = doc[id_field]
                    text = doc[text_field]

                    if text and doc_id:
                        docs[doc_id] = text

            except json.JSONDecodeError:
                continue

    return docs


def write_entity_results(results: List[str], output_file: str) -> None:
    """
    Write entity linking results to file.

    Args:
        results: List of JSON strings containing entity results
        output_file: Path to output file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(result + '\n')


# Example usage functions
def link_documents_from_file(doc_file: str,
                             output_file: str,
                             gcube_token: str,
                             num_workers: int = 4,
                             max_docs: Optional[int] = None) -> None:
    """
    Convenience function to link entities in documents from a file.

    Args:
        doc_file: Path to input document file (JSONL)
        output_file: Path to output file
        gcube_token: GCUBE token for WAT API
        num_workers: Number of parallel workers
        max_docs: Maximum number of documents to process
    """
    # Initialize entity linker
    linker = WATEntityLinker(gcube_token)

    # Process the file
    linker.process_jsonl_file(
        input_file=doc_file,
        output_file=output_file,
        num_workers=num_workers,
        max_docs=max_docs
    )