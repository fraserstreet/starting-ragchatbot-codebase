import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

# Create figure and axis
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.axis('off')

# Define colors
colors = {
    'frontend': '#E3F2FD',
    'api': '#F3E5F5', 
    'rag': '#E8F5E8',
    'ai': '#FFF3E0',
    'vector': '#F1F8E9',
    'tools': '#FCE4EC',
    'session': '#E0F2F1'
}

# Title
ax.text(8, 11.5, 'RAG Chatbot: Query to Response Flow', 
        fontsize=20, fontweight='bold', ha='center')

# 1. Frontend (User Interface)
frontend_box = FancyBboxPatch((0.5, 9.5), 3, 1.5, 
                             boxstyle="round,pad=0.1", 
                             facecolor=colors['frontend'], 
                             edgecolor='blue', linewidth=2)
ax.add_patch(frontend_box)
ax.text(2, 10.6, 'Frontend', fontsize=12, fontweight='bold', ha='center')
ax.text(2, 10.2, 'script.js', fontsize=10, ha='center')
ax.text(2, 9.8, 'User types query', fontsize=9, ha='center')

# Arrow 1: Frontend to FastAPI
arrow1 = ConnectionPatch((3.5, 10.2), (5, 10.2), "data", "data",
                        arrowstyle="->", shrinkA=5, shrinkB=5, 
                        mutation_scale=20, fc="red", lw=2)
ax.add_artist(arrow1)
ax.text(4.2, 10.4, 'POST /api/query', fontsize=8, ha='center', color='red')

# 2. FastAPI Endpoint
api_box = FancyBboxPatch((5, 9.5), 3, 1.5,
                        boxstyle="round,pad=0.1",
                        facecolor=colors['api'],
                        edgecolor='purple', linewidth=2)
ax.add_patch(api_box)
ax.text(6.5, 10.6, 'FastAPI', fontsize=12, fontweight='bold', ha='center')
ax.text(6.5, 10.2, 'app.py', fontsize=10, ha='center')
ax.text(6.5, 9.8, '/api/query endpoint', fontsize=9, ha='center')

# Arrow 2: FastAPI to RAG System
arrow2 = ConnectionPatch((6.5, 9.5), (6.5, 8.5), "data", "data",
                        arrowstyle="->", shrinkA=5, shrinkB=5,
                        mutation_scale=20, fc="red", lw=2)
ax.add_artist(arrow2)
ax.text(7.2, 9, 'rag_system.query()', fontsize=8, ha='left', color='red')

# 3. RAG System Orchestrator
rag_box = FancyBboxPatch((4.5, 7), 4, 1.5,
                        boxstyle="round,pad=0.1",
                        facecolor=colors['rag'],
                        edgecolor='green', linewidth=2)
ax.add_patch(rag_box)
ax.text(6.5, 8.1, 'RAG System', fontsize=12, fontweight='bold', ha='center')
ax.text(6.5, 7.7, 'rag_system.py', fontsize=10, ha='center')
ax.text(6.5, 7.3, 'Orchestrates components', fontsize=9, ha='center')

# 4. Session Manager (left branch)
session_box = FancyBboxPatch((0.5, 5.5), 2.5, 1,
                           boxstyle="round,pad=0.1",
                           facecolor=colors['session'],
                           edgecolor='teal', linewidth=2)
ax.add_patch(session_box)
ax.text(1.75, 6.2, 'Session Manager', fontsize=10, fontweight='bold', ha='center')
ax.text(1.75, 5.8, 'Get history', fontsize=8, ha='center')

# Arrow 3a: RAG to Session Manager
arrow3a = ConnectionPatch((4.5, 7.5), (3, 6), "data", "data",
                         arrowstyle="->", shrinkA=5, shrinkB=5,
                         mutation_scale=20, fc="blue", lw=1.5)
ax.add_artist(arrow3a)

# Arrow 3b: RAG to AI Generator
arrow3b = ConnectionPatch((6.5, 7), (6.5, 5.5), "data", "data",
                         arrowstyle="->", shrinkA=5, shrinkB=5,
                         mutation_scale=20, fc="red", lw=2)
ax.add_artist(arrow3b)
ax.text(7.2, 6.2, 'generate_response()', fontsize=8, ha='left', color='red')

# 5. AI Generator (Claude)
ai_box = FancyBboxPatch((4.5, 4), 4, 1.5,
                       boxstyle="round,pad=0.1",
                       facecolor=colors['ai'],
                       edgecolor='orange', linewidth=2)
ax.add_patch(ai_box)
ax.text(6.5, 5.1, 'AI Generator', fontsize=12, fontweight='bold', ha='center')
ax.text(6.5, 4.7, 'ai_generator.py', fontsize=10, ha='center')
ax.text(6.5, 4.3, 'Claude API + Tools', fontsize=9, ha='center')

# Arrow 4: AI Generator to Tool Manager
arrow4 = ConnectionPatch((8.5, 4.7), (10, 4.7), "data", "data",
                        arrowstyle="->", shrinkA=5, shrinkB=5,
                        mutation_scale=20, fc="red", lw=2)
ax.add_artist(arrow4)
ax.text(9.2, 5, 'Tool execution', fontsize=8, ha='center', color='red')

# 6. Tool Manager
tool_box = FancyBboxPatch((10, 4), 3, 1.5,
                         boxstyle="round,pad=0.1",
                         facecolor=colors['tools'],
                         edgecolor='magenta', linewidth=2)
ax.add_patch(tool_box)
ax.text(11.5, 5.1, 'Tool Manager', fontsize=12, fontweight='bold', ha='center')
ax.text(11.5, 4.7, 'search_tools.py', fontsize=10, ha='center')
ax.text(11.5, 4.3, 'CourseSearchTool', fontsize=9, ha='center')

# Arrow 5: Tool Manager to Vector Store
arrow5 = ConnectionPatch((11.5, 4), (11.5, 2.5), "data", "data",
                        arrowstyle="->", shrinkA=5, shrinkB=5,
                        mutation_scale=20, fc="red", lw=2)
ax.add_artist(arrow5)
ax.text(12.2, 3.2, 'vector_store.search()', fontsize=8, ha='left', color='red')

# 7. Vector Store (ChromaDB)
vector_box = FancyBboxPatch((9.5, 1), 4, 1.5,
                           boxstyle="round,pad=0.1",
                           facecolor=colors['vector'],
                           edgecolor='darkgreen', linewidth=2)
ax.add_patch(vector_box)
ax.text(11.5, 2.1, 'Vector Store', fontsize=12, fontweight='bold', ha='center')
ax.text(11.5, 1.7, 'ChromaDB + Embeddings', fontsize=10, ha='center')
ax.text(11.5, 1.3, 'Semantic search', fontsize=9, ha='center')

# Return arrows (right side)
# Vector Store back to Tool Manager
arrow_back1 = ConnectionPatch((9.5, 1.7), (8.5, 1.7), "data", "data",
                             arrowstyle="->", shrinkA=5, shrinkB=5,
                             mutation_scale=20, fc="green", lw=1.5)
ax.add_artist(arrow_back1)
ax.text(9, 1.9, 'Search results', fontsize=8, ha='center', color='green')

# Connect back to Tool Manager
arrow_back1_up = ConnectionPatch((8.5, 1.7), (8.5, 4.2), "data", "data",
                                arrowstyle="->", shrinkA=5, shrinkB=5,
                                mutation_scale=20, fc="green", lw=1.5)
ax.add_artist(arrow_back1_up)

# Tool Manager back to AI Generator
arrow_back2 = ConnectionPatch((10, 4.2), (8.5, 4.2), "data", "data",
                             arrowstyle="->", shrinkA=5, shrinkB=5,
                             mutation_scale=20, fc="green", lw=1.5)
ax.add_artist(arrow_back2)
ax.text(9.2, 4.4, 'Tool results', fontsize=8, ha='center', color='green')

# AI Generator back to RAG System
arrow_back3 = ConnectionPatch((6.5, 4), (6.5, 7), "data", "data",
                             arrowstyle="->", shrinkA=5, shrinkB=5,
                             mutation_scale=20, fc="green", lw=1.5)
ax.add_artist(arrow_back3)
ax.text(5.8, 5.5, 'Generated\nresponse', fontsize=8, ha='center', color='green')

# RAG System back to FastAPI
arrow_back4 = ConnectionPatch((4.5, 7.7), (3.5, 7.7), "data", "data",
                             arrowstyle="->", shrinkA=5, shrinkB=5,
                             mutation_scale=20, fc="green", lw=1.5)
ax.add_artist(arrow_back4)

# Connect to FastAPI
arrow_back4_up = ConnectionPatch((3.5, 7.7), (3.5, 10), "data", "data",
                                arrowstyle="->", shrinkA=5, shrinkB=5,
                                mutation_scale=20, fc="green", lw=1.5)
ax.add_artist(arrow_back4_up)

# FastAPI back to Frontend
arrow_back5 = ConnectionPatch((5, 9.8), (3.5, 9.8), "data", "data",
                             arrowstyle="->", shrinkA=5, shrinkB=5,
                             mutation_scale=20, fc="green", lw=2)
ax.add_artist(arrow_back5)
ax.text(4.2, 9.6, 'JSON response', fontsize=8, ha='center', color='green')

# Add step numbers
steps = [
    (1.2, 10.8, '1'),
    (6.8, 10.8, '2'), 
    (6.8, 8.3, '3'),
    (6.8, 5.3, '4'),
    (11.8, 5.3, '5'),
    (11.8, 2.3, '6')
]

for x, y, num in steps:
    circle = plt.Circle((x, y), 0.15, color='red', zorder=10)
    ax.add_patch(circle)
    ax.text(x, y, num, fontsize=10, fontweight='bold', ha='center', va='center', color='white')

# Add legend
legend_y = 0.5
ax.text(0.5, legend_y, 'Flow Steps:', fontsize=12, fontweight='bold')
ax.text(0.5, legend_y-0.3, '1. User submits query via frontend', fontsize=9)
ax.text(0.5, legend_y-0.6, '2. FastAPI receives POST request', fontsize=9)
ax.text(0.5, legend_y-0.9, '3. RAG system orchestrates processing', fontsize=9)
ax.text(8, legend_y-0.3, '4. Claude API processes with tools', fontsize=9)
ax.text(8, legend_y-0.6, '5. Search tool queries vector database', fontsize=9)
ax.text(8, legend_y-0.9, '6. ChromaDB performs semantic search', fontsize=9)

# Add data flow indicators
ax.text(14, 8, 'Data Flow:', fontsize=12, fontweight='bold', color='red')
ax.text(14, 7.6, '→ Request flow', fontsize=10, color='red')
ax.text(14, 7.2, '→ Response flow', fontsize=10, color='green')

plt.title('RAG System Query Processing Flow', pad=20)
plt.tight_layout()
plt.savefig('/home/fraser/Tutorials/DeepLearning/Claude_Code/starting-ragchatbot-codebase/query_flow_diagram.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("Query flow diagram saved as 'query_flow_diagram.png'")