from typing import List, Dict
from generation.qa_engine import QAEngine

class EvaluationSuite:
    """Benchmark RAG system performance"""
    
    def __init__(self, qa_engine: QAEngine):
        self.qa_engine = qa_engine
        self.metrics = {
            'accuracy': [],
            'retrieval_times': [],
            'generation_times': [],
            'faithfulness': []
        }
    
    def evaluate_queries(self, test_queries: List[Dict[str, str]]):
        """
        test_queries format:
        [
            {
                'question': 'What is the GDP forecast?',
                'expected_answer': 'The GDP forecast is 2.8%',
                'modalities': ['text', 'table']
            }
        ]
        """
        results = []
        
        for test in test_queries:
            result = self.qa_engine.generate_answer(test['question'])
            
            # Calculate metrics
            accuracy = self._calculate_accuracy(
                result['answer'], 
                test['expected_answer']
            )
            
            results.append({
                'question': test['question'],
                'accuracy': accuracy,
                'retrieval_time': result['retrieval_time_ms'],
                'generation_time': result['generation_time_ms'],
                'modalities_used': [s['type'] for s in result['sources']]
            })
        
        return results
    
    def _calculate_accuracy(self, generated: str, expected: str) -> float:
        """Simple accuracy metric (can be enhanced with BLEU, ROUGE, etc.)"""
        generated_tokens = set(generated.lower().split())
        expected_tokens = set(expected.lower().split())
        
        if not expected_tokens:
            return 0.0
        
        overlap = len(generated_tokens & expected_tokens)
        return overlap / len(expected_tokens)
