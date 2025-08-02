"""Needle in haystack dataset with accurate token counting."""

import os
from typing import List, Optional

import datasets
import numpy as np


_CITATION = """\
@misc{needleinhaystack2024,
  title={Needle in a Haystack - Pressure Testing LLMs},
  author={Greg Kamradt},
  year={2024},
  url={https://github.com/gkamradt/LLMTest_NeedleInAHaystack}
}
"""

_DESCRIPTION = "Needle in haystack evaluation with accurate token counting."
_HOMEPAGE = "https://github.com/gkamradt/LLMTest_NeedleInAHaystack"
_LICENSE = "MIT"


class NeedleHaystackConfig(datasets.BuilderConfig):
    """Configuration for needle haystack dataset."""
    
    def __init__(
        self,
        context_lengths: Optional[List[int]] = None,
        document_depth_percents: Optional[List[int]] = None,
        needle: str = "The secret key is 42 alpha bravo.",
        retrieval_question: str = "What is the secret key?",
        **kwargs
    ):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)
        self.context_lengths = context_lengths or [250, 500, 1000, 2000]
        self.document_depth_percents = document_depth_percents or [0, 25, 50, 75, 100]
        self.needle = needle
        self.retrieval_question = retrieval_question


class NeedleHaystack(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        NeedleHaystackConfig(
            name="needle_haystack",
            description="Needle in haystack with accurate token counting",
        ),
        NeedleHaystackConfig(
            name="needle_haystack_proper",
            description="Needle in haystack with accurate token counting",
        )
    ]

    def _info(self):
        features = datasets.Features(
            {
                "context": datasets.Value("string"),
                "question": datasets.Value("string"), 
                "needle": datasets.Value("string"),
                "context_length": datasets.Value("int32"),
                "depth_percent": datasets.Value("float32"),
                "answer": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"split": "test"},
            ),
        ]

    def _generate_examples(self, split):
        """Generate properly sized needle in haystack examples."""
        config = self.config
        
        # Load haystack text
        haystack_text = self._load_haystack_text()
        
        # Initialize tokenizer for accurate token counting
        tokenizer = self._get_tokenizer()
        
        example_id = 0
        for target_tokens in config.context_lengths:
            for depth_percent in config.document_depth_percents:
                # Generate context with needle inserted
                context, actual_tokens = self._generate_context_with_needle(
                    haystack_text, 
                    config.needle,
                    target_tokens, 
                    depth_percent,
                    tokenizer
                )
                
                yield example_id, {
                    "context": context,
                    "question": config.retrieval_question,
                    "needle": config.needle,
                    "context_length": actual_tokens,
                    "depth_percent": float(depth_percent),
                    "answer": config.needle.strip(),
                }
                example_id += 1

    def _get_tokenizer(self):
        """Get tokenizer for accurate token counting."""
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            return tokenizer
        except:
            # Fallback: approximate tokenization
            return None

    def _count_tokens(self, text: str, tokenizer) -> int:
        """Count tokens accurately."""
        if tokenizer:
            return len(tokenizer.encode(text))
        else:
            # Rough approximation if no tokenizer available
            return len(text.split()) * 1.3  # Account for subword tokenization

    def _load_haystack_text(self) -> str:
        """Load comprehensive background text."""
        # Create rich, varied background text for realistic testing
        background_texts = [
            """In the rapidly evolving world of technology, artificial intelligence has become 
            a cornerstone of modern innovation. Machine learning algorithms are now being 
            deployed across various industries, from healthcare to finance, transforming 
            how we approach complex problems. Natural language processing has particularly 
            seen remarkable advancement, enabling computers to understand and generate 
            human-like text with unprecedented accuracy.""",
            
            """Software development practices have also evolved significantly. Agile methodologies 
            have become the standard, promoting iterative development and continuous 
            improvement. Version control systems like Git have revolutionized collaboration 
            among developers, making it easier to manage code changes and coordinate team efforts.
            Cloud computing has democratized access to powerful computing resources.""",
            
            """Cybersecurity remains a critical concern as our digital footprint continues to 
            expand. Data breaches and cyber attacks have become more sophisticated, requiring 
            organizations to implement robust security measures. Encryption technologies 
            and zero-trust architectures are becoming essential components of any security strategy.
            The rise of remote work has further emphasized the importance of secure access.""",
            
            """User experience design has gained prominence as companies recognize the importance 
            of creating intuitive and engaging interfaces. The focus has shifted from purely 
            functional software to applications that provide delightful user experiences. 
            Mobile-first design principles have become crucial as smartphone usage continues 
            to dominate internet access patterns.""",
            
            """Data science and analytics have become indispensable tools for decision-making. 
            Organizations are leveraging big data to gain insights into customer behavior, 
            optimize operations, and predict future trends. The democratization of data 
            visualization tools has made it easier for non-technical stakeholders to 
            understand and act on data-driven insights.""",
            
            """Quantum computing and blockchain promise to reshape our understanding of 
            computational possibilities. While still in early stages, these technologies 
            hold immense potential for solving problems that are currently intractable 
            with classical computing methods. Research in these areas continues to accelerate
            with significant investments from both private and public sectors."""
        ]
        
        # Repeat and combine to create a large corpus
        full_text = " ".join(background_texts * 50)  # Create a large corpus
        return full_text

    def _generate_context_with_needle(
        self, 
        haystack_text: str, 
        needle: str, 
        target_tokens: int, 
        depth_percent: float,
        tokenizer
    ) -> tuple[str, int]:
        """Generate context with needle inserted at specified depth with accurate token counting."""
        
        # Tokenize the needle
        needle_tokens = self._count_tokens(needle, tokenizer)
        
        # Target tokens for haystack (excluding needle)
        target_haystack_tokens = target_tokens - needle_tokens
        
        # Split haystack into sentences for better insertion points
        sentences = [s.strip() + "." for s in haystack_text.split(".") if s.strip()]
        
        # Build context up to target token count
        context_sentences = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence, tokenizer)
            if current_tokens + sentence_tokens <= target_haystack_tokens:
                context_sentences.append(sentence)
                current_tokens += sentence_tokens
            else:
                # Add partial sentence if needed to reach target
                remaining_tokens = target_haystack_tokens - current_tokens
                if remaining_tokens > 10:  # Only if substantial tokens remain
                    words = sentence.split()
                    partial_sentence = ""
                    for word in words:
                        test_partial = partial_sentence + " " + word if partial_sentence else word
                        if self._count_tokens(test_partial, tokenizer) <= remaining_tokens:
                            partial_sentence = test_partial
                        else:
                            break
                    if partial_sentence:
                        context_sentences.append(partial_sentence)
                break
        
        # Calculate insertion point based on depth percentage
        if depth_percent == 0:
            # Insert at beginning
            final_sentences = [needle] + context_sentences
        elif depth_percent == 100:
            # Insert at end
            final_sentences = context_sentences + [needle]
        else:
            # Insert at specified depth
            insertion_point = int(len(context_sentences) * (depth_percent / 100))
            final_sentences = (context_sentences[:insertion_point] + 
                             [needle] + 
                             context_sentences[insertion_point:])
        
        # Join and get final context
        final_context = " ".join(final_sentences)
        actual_tokens = self._count_tokens(final_context, tokenizer)
        
        return final_context, int(actual_tokens)