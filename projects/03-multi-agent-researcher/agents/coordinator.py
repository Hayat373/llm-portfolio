from .researcher import ResearcherAgent
from .writer import WriterAgent

class Coordinator:
    def __init__(self):
        self.researcher = ResearcherAgent()
        self.writer = WriterAgent()

    def process_query(self, query):
        # Researcher gathers information
        research = self.researcher.research(query)
        
        # Writer creates final response
        final_answer = self.writer.write(research, query)
        
        return final_answer