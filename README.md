# Intelligent Course Planner Dashboard

A Streamlit-powered academic planning dashboard that leverages a Neo4j knowledge graph to visualize student progress, suggest optimal course paths, and provide insightful academic analytics.

---

## Features

- Interactive student dashboard with summaries and visualizations
- Dynamic academic knowledge graph using Neo4j + GDS
- Course prerequisite and co-requisite visualization
- Automated plan generation 

---

## Requirements

- Python 3.10
- Neo4j Desktop or Aura instance with [Graph Data Science (GDS) Plugin](https://neo4j.com/docs/graph-data-science/current/installation/)
- Required Python libraries (see `requirements.txt`)

---

## Project Structure

home.py                        # Main Streamlit dashboard app
pages/
plan_generation.py         # Course plan generation interface
helper.py                      # Core graph logic and utility functions
setup.py                       # Graph database setup and data ingestion
.env                           # Environment variables (Neo4j, OpenRouter)
requirements.txt               # Required Python packages
README.md                      # You're here!
```

---

## Setup Instructions

### 1. Install Neo4j & Graph Data Science

- Download [Neo4j Desktop](https://neo4j.com/download/)
- Create a **local database**
- Enable the **Graph Data Science** plugin via the *Plugins* tab

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Excel Data Files

Place the 11 provided `.xlsx` files in the `data/` directory.

> You may optionally connect to a live database.

### 4. Configure Environment

Create a `.env` file at the root with the following keys:

```ini
NEO4J_HOST=bolt://localhost:7687
NEO4J_PASS=your_password_here
OPENROUTER_KEY=your_openrouter_api_key
```

### 5. Run Setup Script

This will:
- Load data into the Neo4j graph
- Create relationships between nodes
- Build a student-course knowledge graph

```bash
python setup.py
```

### 6. Launch the Dashboard

```bash
streamlit run home.py
```

---

## âš™ï¸ Built-in Assumptions

- Course level is inferred from the course code (e.g., `CS311` â†’ level 3)
- Certain rules are hardcoded (e.g., number of electives, internship placement)
```python
if student_major == 'CS': 
    major_elective_limit = 4
else: 
    major_elective_limit = 3
```

---

## ðŸ§  Future Improvements

- Integration with live student info systems
- Real-time academic alerts
- Advisor collaboration tools
- Improved LLM-driven plan recommendations

---

## ðŸ™Œ Acknowledgments

Built for academic planning and student success by combining the power of graphs and AI. Contributions are welcome!
