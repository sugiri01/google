import nltk
from nltk.tokenize import sent_tokenize
import google.generativeai as genai
import time
import json
import os
from graphviz import Digraph
from flask import Flask, request, jsonify, render_template, send_from_directory

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

# Set up Google Gemini API
genai.configure(api_key='AIzaSyDuUvX_FlvDVveb90NLSga3eEqhnrCdZWA')  # Replace with your actual Gemini API key
model = genai.GenerativeModel('gemini-pro')

class EnhancedFlexibleConceptMapper:
    def __init__(self):
        self.main_graph = Digraph(comment='Main Concept Map')
        self.blooms_graphs = {}
        self.subconceptual_graphs = {}
        self.blooms_levels = self.get_blooms_levels()

    def generate_gemini_response(self, prompt, max_tokens=1000):
        try:
            response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0.2
            ))
            return response.text
        except Exception as e:
            print(f"Error in getting Gemini response: {e}")
            if "Rate limit" in str(e):
                print("Rate limit exceeded. Waiting for 60 seconds before retrying...")
                time.sleep(60)
                return self.generate_gemini_response(prompt, max_tokens)
            return None

    def get_blooms_levels(self):
        prompt = "List the six levels of Bloom's Taxonomy in order of cognitive complexity, from lowest to highest."
        response = self.generate_gemini_response(prompt)
        levels = response.strip().split('\n') if response else ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
        return [level.strip() for level in levels if level.strip()]

    def extract_key_concepts(self, text):
        prompt = f"""
        Analyze the following text and extract key concepts:
        
        {text}
        
        Provide the output as a JSON object with the following structure:
        {{
            "main_concept": "Main Topic",
            "subconcepts": [
                {{
                    "name": "Subconcept 1",
                    "blooms_level": "<Appropriate Bloom's level>",
                    "children": ["Child 1", "Child 2"]
                }},
                {{
                    "name": "Subconcept 2",
                    "blooms_level": "<Appropriate Bloom's level>",
                    "children": ["Child 3", "Child 4"]
                }},
                ...
            ]
        }}
        Ensure there are exactly 6 subconcepts, one for each of the six Bloom's Taxonomy levels.
        Each subconcept should have 2-3 children.
        Use the following Bloom's levels: {', '.join(self.blooms_levels)}
        """
        response = self.generate_gemini_response(prompt)
        if response:
            try:
                json_str = response.strip().strip('`').strip()
                if json_str.startswith('json'):
                    json_str = json_str[4:].strip()
                concepts = json.loads(json_str)
                return concepts
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                print(f"Response was: {response}")
        print("Failed to extract key concepts.")
        return None

    def extract_subconceptual_map(self, subconcept):
        prompt = f"""
        Create a detailed mind map for the concept: {subconcept}
        
        Provide the output as a JSON object with the following structure:
        {{
            "main_concept": "{subconcept}",
            "subconcepts": [
                {{
                    "name": "Related Concept 1",
                    "children": ["Subpoint 1", "Subpoint 2"]
                }},
                {{
                    "name": "Related Concept 2",
                    "children": ["Subpoint 3", "Subpoint 4"]
                }},
                ...
            ]
        }}
        Include 4-6 related concepts, each with 2-3 subpoints.
        """
        response = self.generate_gemini_response(prompt)
        if response:
            try:
                json_str = response.strip().strip('`').strip()
                if json_str.startswith('json'):
                    json_str = json_str[4:].strip()
                subconceptual_map = json.loads(json_str)
                return subconceptual_map
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON for subconceptual map: {e}")
                print(f"Response was: {response}")
        print(f"Failed to generate subconceptual map for {subconcept}.")
        return None

    def build_graphs(self, concepts):
        if not concepts:
            return
        
        def escape_label(label):
            words = label.split()
            lines = []
            current_line = []
            current_length = 0
            for word in words:
                if current_length + len(word) > 20:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = len(word)
                else:
                    current_line.append(word)
                    current_length += len(word) + 1
            if current_line:
                lines.append(' '.join(current_line))
            wrapped_label = '\\n'.join(lines)
            return wrapped_label.replace('"', '\\"').replace('|', '\\|').replace('{', '\\{').replace('}', '\\}').replace('<', '\\<').replace('>', '\\>')
        
        self.main_graph.attr(rankdir='LR', size='8,5')
        self.main_graph.attr('node', shape='box', style='filled', fillcolor='lightblue')
        
        self.main_graph.node(escape_label(concepts['main_concept']), escape_label(concepts['main_concept']))
        for subconcept in concepts['subconcepts']:
            self.main_graph.node(escape_label(subconcept['name']), escape_label(subconcept['name']))
            self.main_graph.edge(escape_label(concepts['main_concept']), escape_label(subconcept['name']))
            
            level = subconcept['blooms_level']
            if level not in self.blooms_graphs:
                self.blooms_graphs[level] = Digraph(comment=f"Bloom's Level: {level}")
                self.blooms_graphs[level].attr(rankdir='LR', size='8,5')
                self.blooms_graphs[level].attr('node', shape='box', style='filled', fillcolor='lightgreen')
                self.blooms_graphs[level].node(escape_label(level), escape_label(level))
            
            self.blooms_graphs[level].node(escape_label(subconcept['name']), escape_label(subconcept['name']))
            self.blooms_graphs[level].edge(escape_label(level), escape_label(subconcept['name']))
            for child in subconcept['children']:
                self.blooms_graphs[level].node(escape_label(child), escape_label(child))
                self.blooms_graphs[level].edge(escape_label(subconcept['name']), escape_label(child))

            # Generate and build subconceptual graph
            subconceptual_map = self.extract_subconceptual_map(subconcept['name'])
            if subconceptual_map:
                subgraph = Digraph(comment=f"Subconceptual Map: {subconcept['name']}")
                subgraph.attr(rankdir='LR', size='8,5')
                subgraph.attr('node', shape='box', style='filled', fillcolor='lightyellow')
                subgraph.node(escape_label(subconceptual_map['main_concept']), escape_label(subconceptual_map['main_concept']))
                for sub in subconceptual_map['subconcepts']:
                    subgraph.node(escape_label(sub['name']), escape_label(sub['name']))
                    subgraph.edge(escape_label(subconceptual_map['main_concept']), escape_label(sub['name']))
                    for child in sub['children']:
                        subgraph.node(escape_label(child), escape_label(child))
                        subgraph.edge(escape_label(sub['name']), escape_label(child))
                self.subconceptual_graphs[subconcept['name']] = subgraph

    def visualize_graph(self, graph, filename):
        graph.render(filename, format='png', cleanup=True)
        print(f"Mind Map saved as '{filename}.png'")
        return f'{filename}.png'

    def generate_maps(self, text):
        concepts = self.extract_key_concepts(text)
        if not concepts:
            print("Failed to generate concepts. Cannot create maps.")
            return

        self.build_graphs(concepts)
        
        concept_maps = {
            'overview': self.visualize_graph(self.main_graph, 'concept_overview_map'),
            'bloom_levels': {},
            'detailed': {}
        }
        
        for level, graph in self.blooms_graphs.items():
            concept_maps['bloom_levels'][level] = self.visualize_graph(graph, f'concepts_{level.lower()}_map')

        for subconcept, graph in self.subconceptual_graphs.items():
            concept_maps['detailed'][subconcept] = self.visualize_graph(graph, f'detailed_{subconcept.lower().replace(" ", "_")}_map')

        return concept_maps

    def summarize_maps(self):
        if not self.main_graph:
            print("No maps to summarize.")
            return "No summary available."

        main_concept = self.main_graph.comment.split(': ')[-1]
        main_summary = self.summarize_single_map(self.main_graph, f"Main {main_concept} Concept Map")
        level_summaries = [self.summarize_single_map(self.blooms_graphs[level], f"Bloom's Level: {level}") 
                           for level in self.blooms_levels if level in self.blooms_graphs]
        subconceptual_summaries = [self.summarize_single_map(graph, f"Detailed Map: {subconcept}")
                                   for subconcept, graph in self.subconceptual_graphs.items()]
        
        full_summary = (f"{main_concept} Concepts Overview:\n\n" + main_summary + 
                        "\n\nBloom's Taxonomy Level Summaries:\n\n" + "\n\n".join(level_summaries) +
                        "\n\nDetailed Subconceptual Map Summaries:\n\n" + "\n\n".join(subconceptual_summaries))
        
        full_summary = full_summary.replace('**', '').replace('*', '').strip()
        
        with open('concept_summary.txt', 'w') as f:
            f.write(full_summary)
        print("Summary saved as 'concept_summary.txt'")
        
        return full_summary

    def summarize_single_map(self, graph, map_title):
        nodes = [node for node in graph.body if node.startswith('	"')]
        edges = [edge for edge in graph.body if '->' in edge]
        
        prompt = f"""
        Analyze the following mind map represented as a graph:
        
        Map Title: {map_title}
        Nodes: {nodes}
        Edges: {edges}
        
        Provide a summary of the main ideas and their relationships based on this mind map.
        If this is a Bloom's Taxonomy level, explain how the concepts relate to that level of thinking.
        The summary should be about 100 words long and highlight the key connections and themes.
        """
        summary = self.generate_gemini_response(prompt, max_tokens=200)
        return summary if summary else f"Failed to generate summary for {map_title}."

app = Flask(__name__, static_folder='static', static_url_path='/static')
mapper = EnhancedFlexibleConceptMapper()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_maps', methods=['POST'])
def generate_maps():
    text = request.json['text']
    concept_maps = mapper.generate_maps(text)
    summary = mapper.summarize_maps()
    
    return jsonify({
        'summary': summary,
        'concept_maps': concept_maps
    })

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('.', filename)

if __name__ == '__main__':
    app.run(debug=True)