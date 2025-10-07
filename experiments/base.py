"""
Base Classes for CF Paper Experiments

Provides the abstract base class for all Conveyance Framework validation experiments,
ensuring consistent interfaces and reusable pipeline logic.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PaperResults:
    """Results from processing a single paper."""
    arxiv_id: str
    title: str
    paper_fetched: bool = False
    code_found: bool = False
    embeddings_generated: bool = False
    stored_in_db: bool = False
    processing_time: float = 0.0
    markdown_words: int = 0
    code_lines: int = 0
    embedding_chunks: int = 0
    error: Optional[str] = None


@dataclass
class ExperimentResults:
    """Overall experiment results."""
    experiment_name: str
    hypothesis: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    papers_results: List[PaperResults] = field(default_factory=list)
    total_papers: int = 0
    successful_papers: int = 0
    repos_found: int = 0
    total_embeddings: int = 0

    def add_paper_result(self, result: PaperResults):
        """
        Record a PaperResults into the experiment and update aggregate counters.
        
        Parameters:
            result (PaperResults): Per-paper results to append. If `result.stored_in_db` is true and `result.error` is None, `successful_papers` is incremented. If `result.code_found` is true, `repos_found` is incremented. `total_embeddings` is increased by `result.embedding_chunks`.
        """
        self.papers_results.append(result)
        if result.stored_in_db and not result.error:
            self.successful_papers += 1
        if result.code_found:
            self.repos_found += 1
        self.total_embeddings += result.embedding_chunks

    def finalize(self):
        """
        Finalize the experiment by recording its end time and total paper count.
        
        Sets `end_time` to the current timestamp and updates `total_papers` to the number of entries in `papers_results`.
        """
        self.end_time = datetime.now()
        self.total_papers = len(self.papers_results)

    @property
    def duration_seconds(self) -> float:
        """
        Return the experiment duration in seconds.
        
        Returns:
            float: Number of seconds between `end_time` and `start_time` if `end_time` is set, otherwise 0.0.
        """
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    @property
    def success_rate(self) -> float:
        """
        Percentage of papers that were successfully stored in the database, expressed as a percentage.
        
        Returns:
            float: Percentage of successful papers; 0.0 if no papers were recorded.
        """
        if self.total_papers == 0:
            return 0.0
        return (self.successful_papers / self.total_papers) * 100


class CFPaperExperiment(ABC):
    """
    Abstract base class for CF paper experiments.

    Provides common pipeline execution logic that all experiments inherit.
    Subclasses must define their paper list and hypothesis.
    """

    @property
    @abstractmethod
    def PAPERS(self) -> List[Tuple[str, str, List[str]]]:
        """
        List of papers the experiment will process.
        
        Returns:
            List[Tuple[str, str, List[str]]]: Tuples of (arxiv_id, title, authors) where `arxiv_id` is the paper identifier, `title` is the paper title, and `authors` is a list of author names.
        """
        pass

    @property
    @abstractmethod
    def HYPOTHESIS(self) -> str:
        """
        Provide the experiment's hypothesis describing the expected range of α.
        
        Returns:
            Hypothesis text describing the expected range of α.
        """
        pass

    @property
    @abstractmethod
    def EXPERIMENT_NAME(self) -> str:
        """
        Experiment identifier used to label this experiment in logs, result objects, and output files.
        
        Returns:
            str: Short, human-readable name that uniquely identifies the experiment (used in logs, saved results, and filenames).
        """
        pass

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the experiment instance with the provided configuration and prepare result tracking.
        
        Parameters:
            config (Dict[str, Any]): Configuration options for the experiment.
        
        Notes:
            Sets up self.results (an ExperimentResults initialized with the class's EXPERIMENT_NAME and HYPOTHESIS)
            and a module-scoped logger named "<module>.<EXPERIMENT_NAME>".
        """
        self.config = config
        self.results = ExperimentResults(
            experiment_name=self.EXPERIMENT_NAME,
            hypothesis=self.HYPOTHESIS
        )
        self.logger = logging.getLogger(f"{__name__}.{self.EXPERIMENT_NAME}")

    @abstractmethod
    def process_paper(self,
                     arxiv_id: str,
                     title: str,
                     authors: List[str]) -> PaperResults:
        """
                     Process a single paper through the experiment pipeline and produce its processing results.
                     
                     Parameters:
                         arxiv_id (str): arXiv identifier of the paper.
                         title (str): Paper title.
                         authors (List[str]): List of author names.
                     
                     Returns:
                         PaperResults: Result object containing processing flags, metrics (timing, word/line counts, embedding chunks), and an optional error message.
                     """
        pass

    def run(self) -> ExperimentResults:
        """
        Run the experiment over all configured papers and aggregate per-paper outcomes.
        
        Processes each configured paper, records each paper's results into the experiment summary, finalizes timing, and returns the aggregated results.
        
        Returns:
            ExperimentResults: Aggregated results for the experiment, including per-paper records, start and end timestamps, and summary metrics.
        """
        self.logger.info(f"Starting {self.EXPERIMENT_NAME} experiment")
        self.logger.info(f"Hypothesis: {self.HYPOTHESIS}")
        self.logger.info(f"Processing {len(self.PAPERS)} papers")

        for i, (arxiv_id, title, authors) in enumerate(self.PAPERS, 1):
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"Processing Paper {i}/{len(self.PAPERS)}: {title} ({arxiv_id})")
            self.logger.info(f"{'='*70}")

            try:
                result = self.process_paper(arxiv_id, title, authors)
                self.results.add_paper_result(result)

                if result.error:
                    self.logger.error(f"Paper {arxiv_id} failed: {result.error}")
                else:
                    self.logger.info(f"Paper {arxiv_id} completed successfully")

            except Exception as e:
                self.logger.exception(f"Unexpected error processing {arxiv_id}")
                result = PaperResults(
                    arxiv_id=arxiv_id,
                    title=title,
                    error=str(e)
                )
                self.results.add_paper_result(result)

        self.results.finalize()
        self._log_summary()
        return self.results

    def _log_summary(self):
        """Log experiment summary statistics."""
        self.logger.info(f"\n{'='*70}")
        self.logger.info("EXPERIMENT SUMMARY")
        self.logger.info(f"{'='*70}")
        self.logger.info(f"Papers Processed:     {self.results.total_papers}")
        self.logger.info(f"Successful:          {self.results.successful_papers}")
        self.logger.info(f"Official Repos Found: {self.results.repos_found}")
        self.logger.info(f"Total Embeddings:     {self.results.total_embeddings}")
        self.logger.info(f"Duration:            {self.results.duration_seconds:.1f} seconds")
        self.logger.info(f"Success Rate:        {self.results.success_rate:.1f}%")

    def save_results(self, output_path: Path):
        """
        Write the experiment's aggregated metadata and per-paper results to a JSON file.
        
        Creates parent directories for the target path if needed and overwrites any existing file at output_path. Timestamps are serialized in ISO 8601 format; the output JSON contains top-level keys: "experiment", "hypothesis", "start_time", "end_time", "duration_seconds", "summary", and "papers".
        
        Parameters:
            output_path (Path): Filesystem path where the JSON results will be written.
        """
        import json

        results_dict = {
            "experiment": self.EXPERIMENT_NAME,
            "hypothesis": self.HYPOTHESIS,
            "start_time": self.results.start_time.isoformat(),
            "end_time": self.results.end_time.isoformat() if self.results.end_time else None,
            "duration_seconds": self.results.duration_seconds,
            "summary": {
                "total_papers": self.results.total_papers,
                "successful_papers": self.results.successful_papers,
                "repos_found": self.results.repos_found,
                "total_embeddings": self.results.total_embeddings,
                "success_rate": self.results.success_rate
            },
            "papers": [
                {
                    "arxiv_id": r.arxiv_id,
                    "title": r.title,
                    "paper_fetched": r.paper_fetched,
                    "code_found": r.code_found,
                    "embeddings_generated": r.embeddings_generated,
                    "stored_in_db": r.stored_in_db,
                    "processing_time": r.processing_time,
                    "markdown_words": r.markdown_words,
                    "code_lines": r.code_lines,
                    "embedding_chunks": r.embedding_chunks,
                    "error": r.error
                }
                for r in self.results.papers_results
            ]
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

        self.logger.info(f"Results saved to {output_path}")