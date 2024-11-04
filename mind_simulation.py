# Configuration
INITIAL_SITUATION = "what is the sum of all natural numbers?"
MAX_ITERATIONS = None  # Set to None for indefinite running, or an integer for fixed iterations
SAVE_DIR = "mind_logs"
THOUGHT_LOG = f"{SAVE_DIR}/thoughts.jsonl"
MEMORY_LOG = f"{SAVE_DIR}/memories.jsonl"
EMBEDDING_LOG = f"{SAVE_DIR}/embeddings.jsonl"
STATE_LOG = f"{SAVE_DIR}/states.jsonl"
QUESTION_LOG = f"{SAVE_DIR}/questions.jsonl"
BELIEF_LOG = f"{SAVE_DIR}/beliefs.jsonl"
SLEEP_DURATION = 2  # Seconds between iterations
CONCLUSION_LOG = f"{SAVE_DIR}/conclusions.jsonl"
CONCLUSION_INTERVAL = 3  # generate conclusion every n iterations

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
import random
import os
import json
import asyncio
from openai import OpenAI
from termcolor import colored
import time
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file


# Core data models
class EmotionalState(str, Enum):
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    CURIOUS = "curious"
    ANXIOUS = "anxious"
    NEUTRAL = "neutral"

class Thought(BaseModel):
    content: str = Field(description="The actual content of the thought")
    source: str = Field(description="Which part of mind generated it")
    intensity: float = Field(description="How strongly this thought presents, between 0 and 1")
    emotion: EmotionalState = Field(description="The emotional state associated with this thought")
    associations: List[str] = Field(description="Associations to other thoughts")

class Question(BaseModel):
    content: str = Field(description="The content of the question")
    source: str = Field(description="Which part of mind generated it")
    importance: float = Field(description="How important is this question, between 0 and 1")
    context: str = Field(description="The context in which this question was generated")

class ConsciousState(BaseModel):
    active_thoughts: List[Thought] = Field(description="Currently active thoughts in consciousness")
    dominant_emotion: EmotionalState = Field(description="The current dominant emotion")
    attention_focus: str = Field(description="What the mind is currently focused on")
    arousal_level: float = Field(description="How aroused the mind is, between 0 and 1")


class Belief(BaseModel):
    statement: str = Field(description="The belief statement")
    confidence: float = Field(description="Confidence level in this belief (0-1)")
    supporting_thoughts: List[str] = Field(description="References to thoughts that support this belief")
    counter_thoughts: List[str] = Field(description="References to thoughts that challenge this belief")
    last_updated: float = Field(description="Timestamp of last update")
    stability: float = Field(description="How stable this belief is (0-1)")

class Conclusion(BaseModel):
    statement: str = Field(description="The conclusive statement")
    confidence: float = Field(description="Confidence in this conclusion (0-1)")
    supporting_beliefs: List[str] = Field(description="Key beliefs that support this conclusion")
    context: str = Field(description="The context in which this conclusion was made")
    timestamp: float = Field(description="When this conclusion was generated")

class MindLogger:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def log_to_file(self, filename: str, data: Dict[str, Any]):
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'a') as f:
            json_str = json.dumps(data)
            f.write(json_str + '\n')
            f.flush()  # Ensure immediate writing to file
        
# Mind Components
class MindComponent:
    def __init__(self, name: str, client: OpenAI):
        self.name = name
        self.client = client

    async def generate_thought(self, context: Dict) -> Thought:
        """Generate a thought based on current context"""
        print(colored(f"\n🤔 {self.name.title()} component generating thought...", "cyan"))
        prompt = self._create_prompt(context)
        print(colored(f"  └─ Using prompt: {prompt}", "cyan", attrs=["dark"]))

        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            response_format=Thought
        )

        thought = completion.choices[0].message.parsed
        print(colored(f"  └─ Generated thought: {thought.content}", "cyan"))
        return thought

    def _get_system_prompt(self) -> str:
        raise NotImplementedError

    def _create_prompt(self, context: Dict) -> str:
        raise NotImplementedError
    
class EmotionalProcessor(MindComponent):
    def _get_system_prompt(self) -> str:
        return """You are the emotional processing center of a mind.
            Generate emotional responses and associated thoughts based on current context."""

    def _create_prompt(self, context: Dict) -> str:
        return f"Given the current situation: {context.get('situation', '')}, " \
               f"and current emotional state: {context.get('current_emotion', '')}, " \
               f"generate an emotional thought response."

class RationalAnalyzer(MindComponent):
    def _get_system_prompt(self) -> str:
        return """You are the rational analysis center of a mind.
            Generate logical thoughts and analytical observations based on current context."""

    def _create_prompt(self, context: Dict) -> str:
        return f"Analyze this situation logically: {context.get('situation', '')}"
    

class QuestionGenerator(MindComponent):
    def _get_system_prompt(self) -> str:
        return """You are the curiosity center of a mind. Generate meaningful questions 
            based on current thoughts, context, and the initial exploration topic. Focus on deep,
            exploratory questions that build upon previous insights."""

    def _create_prompt(self, context: Dict) -> str:
        thoughts = context.get('active_thoughts', [])
        thought_contents = [t.content for t in thoughts] if thoughts else []
        initial_situation = context.get('initial_situation', '')

        return f"""Initial exploration topic: {initial_situation}
            Based on the current thoughts: {thought_contents}
            and situation: {context.get('situation', '')},
            generate a question that helps explore and build upon our understanding of the initial topic."""

    async def generate_question(self, context: Dict) -> Question:
        print(colored("\n❓ Generating new question...", "magenta"))

        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": self._create_prompt(context)}
            ],
            response_format=Question
        )

        question = completion.choices[0].message.parsed
        print(colored(f"  └─ New question: {question.content}", "magenta"))
        return question
    

class MemorySystem(MindComponent):
    def __init__(self, name: str, client: OpenAI, logger: 'MindLogger'):
        super().__init__(name, client)
        self.memories: List[Thought] = []
        self.logger = logger
        self.embeddings_cache: Dict[str, List[float]] = {}
        self._load_existing_memories()

    def _load_existing_memories(self):
        """Load existing memories and embeddings from files"""
        try:
            # Load memories
            memory_count = 0
            if os.path.exists(MEMORY_LOG):
                with open(MEMORY_LOG, 'r') as f:
                    for line in f:
                        memory_data = json.loads(line)
                        thought = Thought(**memory_data)
                        self.memories.append(thought)
                        memory_count += 1

            # Load embeddings
            embedding_count = 0
            if os.path.exists(EMBEDDING_LOG):
                with open(EMBEDDING_LOG, 'r') as f:
                    for line in f:
                        embedding_data = json.loads(line)
                        self.embeddings_cache[embedding_data['content']] = embedding_data['embedding']
                        embedding_count += 1

            print(f"Loaded {memory_count} memories and {embedding_count} embeddings")

        except Exception as e:
            print(f"Error loading memories: {e}")
            # Initialize empty if loading fails
            self.memories = []
            self.embeddings_cache = {}

    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI's API"""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0

    async def store_memory(self, thought: Thought):
        """Store memory and its embedding"""
        print(colored(f"\n💭 Storing memory: {thought.content[:50]}...", "yellow"))
        
        # Get embedding for the thought content
        embedding = await self._get_embedding(thought.content)
        self.embeddings_cache[thought.content] = embedding
        
        # Store memory
        self.memories.append(thought)
        memory_data = thought.dict()
        self.logger.log_to_file('memories.jsonl', memory_data)
        
        # Store the embedding
        embedding_data = {
            'content': thought.content,
            'embedding': embedding
        }
        self.logger.log_to_file('embeddings.jsonl', embedding_data)

    async def retrieve_relevant_memories(self, context: Dict, num_memories: int = 3, similarity_threshold: float = 0.5):
        """Retrieve relevant memories based on semantic similarity"""
        print(colored("\n🔍 Searching for relevant memories...", "yellow"))
        if not self.memories:
            return []

        # Create a combined context string
        context_string = f"{context.get('situation', '')} {context.get('current_emotion', '')}"
        if 'active_thoughts' in context:
            thought_contents = [t.content for t in context['active_thoughts']]
            context_string += ' ' + ' '.join(thought_contents)

        print(colored(f"  └─ Context: {context_string[:100]}...", "yellow", attrs=["dark"]))

        # Get embedding for the context
        context_embedding = await self._get_embedding(context_string)

        # Calculate similarities and sort memories
        similarities = []
        for memory in self.memories:
            if memory.content in self.embeddings_cache:
                memory_embedding = self.embeddings_cache[memory.content]
                similarity = self._cosine_similarity(context_embedding, memory_embedding)
                if similarity >= similarity_threshold:
                    similarities.append((memory, similarity))

        # Sort by similarity in descending order and get top N memories
        sorted_memories = sorted(similarities, key=lambda x: x[1], reverse=True)
        relevant_memories = [memory for memory, similarity in sorted_memories[:num_memories]]

        # Print found memories
        if relevant_memories:
            print(colored(f"  └─ Found {len(relevant_memories)} relevant memories", "yellow"))
            for i, memory in enumerate(relevant_memories, 1):
                print(colored(f"    {i}. {memory.content[:100]}...", "yellow", attrs=["dark"]))
        else:
            print(colored("  └─ No relevant memories found", "yellow", attrs=["dark"]))

        return relevant_memories


class BeliefSystem(MindComponent):
    def __init__(self, name: str, client: OpenAI, logger: 'MindLogger'):
        super().__init__(name, client)
        self.beliefs: List[Belief] = []
        self.logger = logger

    def _get_system_prompt(self) -> str:
        return """You are the belief formation center of a mind. Analyze thoughts and form 
        coherent beliefs and conclusions. Consider evidence both for and against each belief."""

    def _create_prompt(self, context: Dict) -> str:
        thoughts = context.get('active_thoughts', [])
        thought_contents = [t.content for t in thoughts] if thoughts else []
        existing_beliefs = [b.statement for b in self.beliefs] if self.beliefs else []

        return f"""Based on these thoughts: {thought_contents}
        And existing beliefs: {existing_beliefs}
        Analyze the evidence and form or update a belief/conclusion."""

    async def evaluate_beliefs(self, context: Dict):
        """Evaluate current thoughts and update beliefs"""
        print(colored("\n🤔 Evaluating beliefs...", "blue"))

        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": self._create_prompt(context)}
            ],
            response_format=Belief
        )

        new_belief = completion.choices[0].message.parsed
        self._update_beliefs(new_belief)
        self._log_belief(new_belief)

    def _update_beliefs(self, new_belief: Belief):
        """Update existing beliefs or add new ones"""
        # Find similar existing belief
        similar_beliefs = [b for b in self.beliefs 
                        if self._belief_similarity(b.statement, new_belief.statement) > 0.8]

        if similar_beliefs:
            # Update existing belief
            existing_belief = similar_beliefs[0]
            # Average confidence based on new evidence
            existing_belief.confidence = (existing_belief.confidence + new_belief.confidence) / 2
            existing_belief.supporting_thoughts.extend(new_belief.supporting_thoughts)
            existing_belief.counter_thoughts.extend(new_belief.counter_thoughts)
            existing_belief.last_updated = time.time()
            # Increase stability with each confirmation
            existing_belief.stability = min(1.0, existing_belief.stability + 0.1)
        else:
            # Add new belief
            new_belief.last_updated = time.time()
            new_belief.stability = 0.1  # Start with low stability
            self.beliefs.append(new_belief)

    def _belief_similarity(self, belief1: str, belief2: str) -> float:
        """Simple similarity check using word overlap"""
        words1 = set(belief1.lower().split())
        words2 = set(belief2.lower().split())
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0

    def _log_belief(self, belief: Belief):
        """Log belief to file"""
        belief_data = belief.dict()
        self.logger.log_to_file('beliefs.jsonl', belief_data)


class ConclusionGenerator(MindComponent):
    def __init__(self, name: str, client: OpenAI, logger: 'MindLogger'):
        super().__init__(name, client)
        self.logger = logger

    def _get_system_prompt(self) -> str:
        return """You are the conclusion forming center of a mind. 
            Analyze current beliefs and thoughts to form high-level conclusions 
            about the topic being explored."""

    def _create_prompt(self, context: Dict) -> str:
        beliefs = context.get('beliefs', [])
        belief_statements = [b.statement for b in beliefs] if beliefs else []
        
        return f"""Based on these beliefs: {belief_statements}
            and the current situation: {context.get('situation', '')},
            form a conclusion about our understanding so far."""

    async def generate_conclusion(self, context: Dict) -> Conclusion:
        print(colored("\n📝 Generating conclusion...", "blue"))
        
        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": self._create_prompt(context)}
            ],
            response_format=Conclusion
        )

        conclusion = completion.choices[0].message.parsed
        self._log_conclusion(conclusion)
        return conclusion

    def _log_conclusion(self, conclusion: Conclusion):
        """Log conclusion to file"""
        conclusion_data = conclusion.dict()
        self.logger.log_to_file('conclusions.jsonl', conclusion_data)


class Mind:
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
        self.logger = MindLogger(SAVE_DIR)
        self.components = {
            'emotional': EmotionalProcessor('emotional', self.client),
            'rational': RationalAnalyzer('rational', self.client),
            'memory': MemorySystem('memory', self.client, self.logger),
            'curiosity': QuestionGenerator('curiosity', self.client),
            'belief': BeliefSystem('belief', self.client, self.logger),
            'conclusion': ConclusionGenerator('conclusion', self.client, self.logger)
        }
        
        self.conscious_state = ConsciousState(
            active_thoughts=[],
            dominant_emotion=EmotionalState.NEUTRAL,
            attention_focus="idle",
            arousal_level=0.5
        )
        
        self.questions: List[Question] = []
        self.initial_situation = None

    def _log_state(self):
        state_data = {
            'dominant_emotion': self.conscious_state.dominant_emotion,
            'attention_focus': self.conscious_state.attention_focus,
            'arousal_level': self.conscious_state.arousal_level,
            'active_thoughts_count': len(self.conscious_state.active_thoughts)
        }
        self.logger.log_to_file('states.jsonl', state_data)

    def _log_thought(self, thought: Thought):
        thought_data = thought.dict()
        self.logger.log_to_file('thoughts.jsonl', thought_data)

    def _log_question(self, question: Question):
        question_data = question.dict()
        self.logger.log_to_file('questions.jsonl', question_data)

    async def _generate_new_question(self) -> str:
        """Generate a new question based on current state"""
        context = {
            'active_thoughts': self.conscious_state.active_thoughts,
            'situation': self.conscious_state.attention_focus,
            'emotion': self.conscious_state.dominant_emotion,
            'initial_situation': self.initial_situation
        }

        question = await self.components['curiosity'].generate_question(context)
        self.questions.append(question)
        self._log_question(question)
        return question.content

    async def process_situation(self, situation: str):
        print(colored(f"\n🤔 Processing situation: {situation}", "magenta", attrs=["bold"]))
        print(colored("-" * 50, "magenta"))

        context = {
            'situation': situation,
            'current_emotion': self.conscious_state.dominant_emotion,
            'arousal_level': self.conscious_state.arousal_level
        }

        print(colored("\n🧠 Current state:", "blue"))
        print(colored(f"  └─ Emotion: {self.conscious_state.dominant_emotion}", "blue"))
        print(colored(f"  └─ Arousal: {self.conscious_state.arousal_level}", "blue"))

        # Generate thoughts from different components
        print(colored("\n💭 Generating component responses...", "green"))
        emotional_thought = await self.components['emotional'].generate_thought(context)
        time.sleep(0.5)  # Add slight delay for readability
        rational_thought = await self.components['rational'].generate_thought(context)

        # Log thoughts
        self._log_thought(emotional_thought)
        self._log_thought(rational_thought)

        # Store thoughts in memory
        print(colored("\n💾 Storing new thoughts in memory...", "yellow"))
        memory_system = self.components['memory']
        await memory_system.store_memory(emotional_thought)
        await memory_system.store_memory(rational_thought)

        # Retrieve relevant past memories
        relevant_memories = await memory_system.retrieve_relevant_memories(
            context,
            num_memories=3,
            similarity_threshold=0.7)

        # Update conscious state
        print(colored("\n🔄 Updating conscious state...", "magenta"))
        self.conscious_state.active_thoughts = [emotional_thought, rational_thought] + relevant_memories
        old_emotion = self.conscious_state.dominant_emotion
        self.conscious_state.dominant_emotion = self._determine_dominant_emotion()
        self.conscious_state.attention_focus = situation
        print(colored("\n🧠 Updated state:", "blue"))
        print(colored(f"  └─ Emotion: {old_emotion} → {self.conscious_state.dominant_emotion}", "blue"))
        print(colored(f"  └─ Attention: {self.conscious_state.attention_focus}", "blue"))
        print(colored(f"  └─ Active thoughts: {len(self.conscious_state.active_thoughts)}", "blue"))
        print(colored("-" * 50 + "\n", "magenta"))

        # Evaluate and update beliefs
        belief_context = {
            'active_thoughts': self.conscious_state.active_thoughts,
            'situation': situation,
            'emotion': self.conscious_state.dominant_emotion
        }
        await self.components['belief'].evaluate_beliefs(belief_context)

    def _determine_dominant_emotion(self) -> EmotionalState:
        """Determine the dominant emotion based on active thoughts"""
        if not self.conscious_state.active_thoughts:
            return EmotionalState.NEUTRAL

        # Count emotion frequencies
        emotion_counts = {}
        max_intensity = 0
        dominant_emotion = EmotionalState.NEUTRAL

        for thought in self.conscious_state.active_thoughts:
            if thought.intensity > max_intensity:
                max_intensity = thought.intensity
                dominant_emotion = thought.emotion
            
            # Track frequency of each emotion
            if thought.emotion in emotion_counts:
                emotion_counts[thought.emotion] += 1
            else:
                emotion_counts[thought.emotion] = 1

        # If there's a tie in intensity, use the most frequent emotion
        max_frequency = 0
        most_frequent_emotion = dominant_emotion
        
        for emotion, count in emotion_counts.items():
            if count > max_frequency:
                max_frequency = count
                most_frequent_emotion = emotion

        # Return the emotion with highest intensity, or most frequent if tied
        return dominant_emotion if max_intensity > 0 else most_frequent_emotion

    async def explore(self, initial_situation: str):
        """Main exploration loop"""
        print(colored(f"\n🔍 Starting exploration of: {initial_situation}", "green", attrs=["bold"]))
        self.initial_situation = initial_situation
        iteration = 0

        while True:
            if MAX_ITERATIONS and iteration >= MAX_ITERATIONS:
                print(colored("\n✨ Reached maximum iterations", "yellow"))
                break

            print(colored(f"\n📍 Iteration {iteration}", "cyan", attrs=["bold"]))
            
            # Process current situation
            current_situation = (self.questions[-1].content 
                               if self.questions 
                               else initial_situation)
            
            await self.process_situation(current_situation)

            # Generate conclusions periodically
            if iteration > 0 and iteration % CONCLUSION_INTERVAL == 0:
                conclusion_context = {
                    'beliefs': self.components['belief'].beliefs,
                    'situation': current_situation
                }
                await self.components['conclusion'].generate_conclusion(conclusion_context)

            # Generate new question for next iteration
            new_question = await self._generate_new_question()
            print(colored(f"\n❓ Next question: {new_question}", "magenta"))

            # Log current state
            self._log_state()

            # Sleep between iterations
            await asyncio.sleep(SLEEP_DURATION)
            iteration += 1


async def main():
    print(colored("\n🧠 Initializing Self-Exploring Mind", "white", "on_blue", attrs=["bold"]))
    print(colored("-" * 50, "blue"))
    print(colored(f"Logs will be saved to: {SAVE_DIR}", "yellow"))

    client = OpenAI()
    mind = Mind(client)
    
    await mind.explore(INITIAL_SITUATION)

if __name__ == "__main__":
    asyncio.run(main())

