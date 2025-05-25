# imports 
from collections import defaultdict
import math
from openai import OpenAI
from py2neo import Graph
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pyvis.network import Network
from dotenv import load_dotenv
import os
from py2neo import Graph
from itertools import combinations
import pandas as pd
from efficient_apriori import apriori as ap
import matplotlib.pyplot as plt
import seaborn as sns

################################################################################################################################################################

# Load environment variables
load_dotenv()

neo4j_host = os.getenv("NEO4J_HOST")
neo4j_password = os.getenv("NEO4J_PASS")

# graph connection 
graph = Graph(neo4j_host, auth=("neo4j",  neo4j_password))

# Validate API key
api_key = os.getenv("OPENROUTER_KEY")
if not api_key:
    raise ValueError("API key not found in environment variables.")

client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

grade_weights = {
    "A": 4, "A-": 3.7, "B+": 3.3, "B": 3, "B-": 2.7,
    "C+": 2.3, "C": 2, "C-": 1.7, "D+": 1.2, "D": 1, "P": 1, "F": 0, "FA": 0, "AW": 0, "NP": 0, "W": 0
}

required_courses = {
    "mandatory": 37,   # Total mandatory courses required
    "major_elective": 0, ## updated later based on student's major 
    "gen_elective_1": 1,
    "gen_elective_2": 1
}

################################################################################################################################################################

def filter_courses_by_major(graph, major_id):
    # Filter courses by major and create relationships
    query = """
        MATCH (m:Major {id: $major_id})-[:has_course]->(c:Course)
        RETURN COUNT(c) AS course_count
    """
    result = graph.run(query, major_id=major_id).data()
    return result[0]["course_count"] > 0

def project_filtered_graph(graph):
    
    query_check = f"""
    CALL gds.graph.exists('majorSubgraph') YIELD exists
    """
    result = graph.run(query_check).evaluate()

    if result:  # If the graph exists, result will be True
        print(f"Graph 'majorSubgraph' already exists. Deleting it before creating a new one.")
        # Delete the existing graph
        query_drop = f"""
        CALL gds.graph.drop('majorSubgraph')
        """
        graph.run(query_drop)
        
    # Project a filtered subgraph
    query = """
        CALL gds.graph.project(
            'majorSubgraph',
            ['Course', 'Student', 'Topic', 'Level', 'Major'],  // Node types to include
            {
                prerequisite_for: { type: 'prerequisite_for', properties: 'weight' },
                co_course: { type: 'co_course', properties: 'weight' },
                requires: { type: 'requires', properties: 'weight' },
                completed_by: { type: 'completed_by', properties: 'weight' },
                //registered_for: { type: 'registered_for', properties: 'weight' },
                material_topic: { type: 'material_topic', properties: 'weight' },
                description_topic: { type: 'description_topic', properties: 'weight' },
                outcome_topic: { type: 'outcome_topic', properties: 'weight' },
                has_level: { type: 'has_level', properties: 'weight' },
                completed: { type: 'completed', properties: 'weight' }
            }
        )
    """
    graph.run(query)
    return True

def get_course_embeddings_for_major(graph, major_id):
    
    filter_courses_by_major(graph, major_id)
    
    project_filtered_graph(graph)
    
    # Generate embeddings using Node2Vec
    query = """
        CALL gds.beta.node2vec.stream(
            'majorSubgraph',
            {
                embeddingDimension: 256,
                walkLength: 60,
                relationshipWeightProperty: 'weight',
                iterations: 20,
                returnFactor: 1.5,
                inOutFactor: 0.3 ,
                nodeLabels: ['Course', 'Topic'] // Apply Node2Vec only to Course nodes
            }
        )
        YIELD nodeId, embedding
        RETURN gds.util.asNode(nodeId).id AS course_id, 
               gds.util.asNode(nodeId).name AS course_name, 
               embedding
    """
    result = graph.run(query)

    # Parse the result
    embeddings = []
    for record in result:
        embeddings.append((record["course_id"], record["course_name"], record["embedding"]))
    return embeddings

################################################################################################################################################################

def get_prerequisites_dict():
    query = """
    MATCH (c1:Course)-[:prerequisite_for]->(c2:Course)
    RETURN c2.id AS dependent_course_id, c1.id AS prerequisite_id
    """
    
    prerequisites_dict = {}

    # Execute the query
    result = graph.run(query)

    for record in result:
        dependent_course = record["dependent_course_id"]
        prerequisite_course = record["prerequisite_id"]
            
        # Add to dictionary
        if dependent_course not in prerequisites_dict:
            prerequisites_dict[dependent_course] = []
            
        prerequisites_dict[dependent_course].append(prerequisite_course)
    
    return prerequisites_dict

def get_cocourses_dict():
    query = """
    MATCH (c1:Course)-[:co_course]->(c2:Course)
    RETURN c2.id AS course_1, c1.id AS course_2
    """
    
    corequisites_dict = {}

    # Execute the query
    result = graph.run(query)

    for record in result:
        course_1 = record["course_1"]
        course_2 = record["course_2"]
            
        # Add to dictionary
        if course_1 not in corequisites_dict:
            corequisites_dict[course_1] = []
            
        corequisites_dict[course_1].append(course_2)
    
    return corequisites_dict

def get_completed_courses(student_id):
    query = """
    MATCH (s:Student {id: $student_id})-[r:completed]->(c:Course)
    RETURN c.id AS course_id
    """
    result = graph.run(query, student_id=student_id)
    return [record["course_id"] for record in result]

def get_registered_courses(student_id):
    query = """
    MATCH (s:Student {id: $student_id})-[r:registered_for]->(c:Course)
    RETURN c.id AS course_id
    """
    result = graph.run(query, student_id=student_id)
    return [record["course_id"] for record in result]

def get_remaining_courses(completed_courses, registered_courses, student_major):
    query = """
    MATCH (c:Course)<-[:has_course]-(m:Major)
    WHERE m.id = $student_major
    AND NOT c.id IN $completed_courses
    AND NOT c.id IN $registered_courses
    RETURN c.id AS course_id, c.type AS type, c.availability AS availability, c.credit_hours AS credit_hours, c.name AS name
    """
    result = graph.run(query, completed_courses=completed_courses, registered_courses=registered_courses, student_major = student_major)
    return [{
        "course_id": record["course_id"],
        "name": record["name"],
        "type": record["type"],
        "availability": record["availability"],
        "credit_hours": record["credit_hours"]
    } for record in result]

def get_all_courses_for_major(student_major):
    query = """
    MATCH (c:Course)<-[:has_course]-(m:Major)
    WHERE m.id = $student_major
    RETURN c.id AS course_id, c.type AS type, c.availability AS availability, 
           c.credit_hours AS credit_hours, c.name AS name
    """
    result = graph.run(query, student_major=student_major)
    return [{
        "course_id": record["course_id"],
        "name": record["name"],
        "type": record["type"],
        "availability": record["availability"],
        "credit_hours": record["credit_hours"]
    } for record in result]

def get_prerequisites(course_id, completed_courses):
    query = """
    MATCH (c:Course {id: $course_id})-[:requires]->(p:Course)
    RETURN COLLECT(p.id) AS prerequisites
    """
    result = graph.run(query, course_id=course_id)
    
    data = result.data()
    
    if data and 'prerequisites' in data[0]:
        prerequisites = data[0]['prerequisites']
    else:
        prerequisites = []
        
    return not prerequisites or all(prereq in completed_courses for prereq in prerequisites) # True if all prerequisites were completed

def get_corequisites(course_id):
    query = """
    MATCH (c:Course {id: $course_id})-[:co_course]->(cr:Course)
    RETURN COLLECT(cr.id) AS corequisites
    """
    result = graph.run(query, course_id=course_id)
    
    data = result.data()
    
    # Initialize corequisites as an empty list if no corequisites are found
    corequisites = data[0]['corequisites'] if data and 'corequisites' in data[0] else []
    
    # If there are no corequisites, return None
    if not corequisites:
        return None
    
    # Return the list of remaining corequisites that are not in the current semester plan or available courses
    return corequisites

def get_next_semester(current_semester):
    return 'fall' if current_semester == 'spring' else 'spring'

def count_completed_courses(course_ids, elective_type):
    if not course_ids:
        return 0
    
    query = """
    MATCH (c:Course)
    WHERE c.id IN $course_ids
    RETURN c.id AS course_id, c.type AS course_type
    """
    result = graph.run(query, course_ids=course_ids)
    
    elective_courses = [record['course_id'] for record in result if record['course_type'] == elective_type]
    
    return len(elective_courses)

def get_major(student_id):
    query = """
    MATCH (m:Major)-[:has_student]-(s:Student) 
    WHERE s.id = $student_id 
    RETURN m.id 
    """
    result = graph.run(query, student_id=student_id)
    
    data = result.data()

    if data:
        return data[0]['m.id']
    return None

def get_course_details(course_id, courses_list):
   
    # Fetch the course details
    course = next((course for course in courses_list if course["course_id"] == course_id), None)
    
    # If the course is found, return its details
    if course:
        return {
            'name': course.get('name', 'Unknown Course'),
            'credit_hours': course.get('credit_hours', 0),
            'priority': course.get('priority_score', '-')
        }
    
    # If the course is not found, return default values
    return {'name': 'Unknown Course', 'credit_hours': 0, 'priority': '-'}

def get_courses_and_relationships(graph, student_id, major_id):
    # Fetch courses the student has completed and registered
    student_courses_query = """
    MATCH (s:Student {id: $student_id})-[r:registered_for|completed]->(c:Course)
    RETURN c.id AS course_id, 
           CASE 
               WHEN type(r) = 'registered_for' THEN 'registered' 
               WHEN type(r) = 'completed' THEN 'completed' 
           END AS status
    """
    student_courses = list(graph.run(student_courses_query, student_id=student_id))  # Convert to list

    # Fetch all courses and relationships for the specified major
    all_courses_query = """
    MATCH (m:Major)-[r:has_course]->(c:Course) 
    WHERE m.id = $major_id 
    RETURN c.id AS course_id, c.name AS course_name
    """
    all_courses = list(graph.run(all_courses_query, major_id=major_id))  # Convert to list

    # Fetch prerequisites that belong to the same major
    prerequisites_query = """
    MATCH (m:Major)-[:has_course]->(c:Course)-[:requires]->(p:Course)
    WHERE m.id = $major_id
    RETURN c.id AS course_id, p.id AS prerequisite_id
    """
    prerequisites = list(graph.run(prerequisites_query, major_id=major_id))  # Convert to list

    # Fetch co-courses that belong to the same major
    co_courses_query = """
    MATCH (m:Major)-[:has_course]->(c:Course)-[:co_course]->(cc:Course)
    WHERE m.id = $major_id
    RETURN c.id AS course_id, cc.id AS co_course_id
    """
    co_courses = list(graph.run(co_courses_query, major_id=major_id))  # Convert to list
    
    return student_courses, all_courses, prerequisites, co_courses

################################################################################################################################################################

def rank_courses_by_embeddings(embeddings_with_ids, remaining_courses, completed_courses_ids):
    
    prerequisites_dict = get_prerequisites_dict()
    
    course_ids = [course_id for course_id, _, _ in embeddings_with_ids]

    course_vectors = np.array([embedding for _, _, embedding in embeddings_with_ids])

    # Precompute course similarities using vectorized operations
    course_similarity_matrix = cosine_similarity(course_vectors)

    # Use remaining courses directly
    remaining_course_ids = [course['course_id'] for course in remaining_courses]
    remaining_course_indices = [course_ids.index(course_id) for course_id in remaining_course_ids]
    
    # Initialize course scores
    course_scores = np.zeros(len(course_ids))
    completed_score = np.zeros(len(course_ids))

    # Calculate importance based on embedding norm
    course_importance = {course_ids[i]: np.log1p(np.linalg.norm(course_vectors[i])) for i in range(len(course_ids))}
       
    # Iterate over remaining courses to calculate their scores
    for i in remaining_course_indices:

        course_id = course_ids[i]
        course_scores[i] = course_importance[course_id] + 1
        
        level = course_id[4]  # Extract the 5th character for level
        if level == '5' : 
            course_scores[i]*= 0.5
        elif level == '4' : 
            course_scores[i]*= 1
        elif level == '3':
            course_scores[i]*= 2
        elif level == '2':
            course_scores[i]*= 5
        else:
            course_scores[i]*= 10

        # Increase score based on completed courses' similarity, if any
        if completed_courses_ids:
            for completed_course_id in completed_courses_ids:
                if completed_course_id in course_ids:
                    completed_index = course_ids.index(completed_course_id)
                    completed_score[i] += course_similarity_matrix[completed_index][i] 

    # Initial ranking of courses by scores
    ranked_courses = sorted(zip(remaining_course_ids, [course_scores[i] for i in remaining_course_indices]), key=lambda x: x[1], reverse=True)

    # Map each remaining course ID to its current rank position for easy lookup
    ranked_course_dict = {course_id: score for course_id, score in ranked_courses}

    # Increase rank for prerequisites of high-ranking courses
    for course_id, score in ranked_courses:
        if course_id in prerequisites_dict:
            for prereq_id in prerequisites_dict[course_id]:
                if prereq_id in ranked_course_dict:
                    ranked_course_dict[prereq_id] += score * 0.5  # Adjust boost factor as needed

    # Re-rank courses with the adjusted scores
    final_ranked_courses = sorted(ranked_course_dict.items(), key=lambda x: x[1], reverse=True)
        
    return final_ranked_courses

################################################################################################################################################################

def dynamically_balance_courses(courses):
    
    # Split courses into high, medium, and low priority (roughly equal thirds)
    third = len(courses) // 3
    high_priority = courses[:third]
    medium_priority = courses[third:2 * third]
    low_priority = courses[2 * third:]

    # Initialize list to hold the final balanced distribution
    balanced_courses = []

    # Proportions for high, medium, and low priority courses in a group
    high_count = 3
    medium_count = 2
    low_count = 1

    # Distribute courses into balanced groups
    while high_priority or medium_priority or low_priority:
        group = []

        # Add high-priority courses
        for _ in range(high_count):
            if high_priority:
                group.append(high_priority.pop(0))

        # Add medium-priority courses
        for _ in range(medium_count):
            if medium_priority:
                group.append(medium_priority.pop(0))

        # Add low-priority courses
        for _ in range(low_count):
            if low_priority:
                group.append(low_priority.pop(0))

        # Add the group to the balanced courses list
        balanced_courses.extend(group)

    
    return balanced_courses

################################################################################################################################################################

def generate_plan(student_id, max_credit_hours, current_registration_semester, courses_list):
    plan = []  

    student_major = get_major(student_id)
    if student_major == 'CS': 
        major_elective_limit = 4
    else: 
        major_elective_limit = 3

    gen_elective_limit = 1

    completed_courses = get_completed_courses(student_id) 
    registered_courses = get_registered_courses(student_id)    
    remaining_courses = get_remaining_courses(completed_courses, registered_courses, student_major)
    completed_courses += get_registered_courses(student_id)

    course_embeddings = get_course_embeddings_for_major(graph, student_major)

    # Calculate priority for each course
    ranked_courses = rank_courses_by_embeddings(course_embeddings, remaining_courses, completed_courses)

    # Create a dictionary of course priorities
    course_priority_dict = {course_id: score for course_id, score in ranked_courses}

    # Assign priority scores directly to courses_list
    for course in courses_list:
        course_id = course['course_id']
        priority_score = course_priority_dict.get(course_id, None)
        course['priority_score'] = priority_score

    # Normalize scores within courses_list
    min_score = min(course['priority_score'] for course in courses_list if course['priority_score'] is not None)
    max_score = max(course['priority_score'] for course in courses_list if course['priority_score'] is not None)

    for course in courses_list:
        if course['priority_score'] is not None:
            course['priority_score'] = (course['priority_score'] - min_score) / (max_score - min_score)

    # Filter remaining_courses from courses_list
    remaining_courses_with_priority = [
        {**course, 'priority_score': course['priority_score']}
        for course in courses_list if course['course_id'] in {c['course_id'] for c in remaining_courses}
    ]

    # Sort the remaining_courses by priority_score in descending order (higher score = higher priority)
    remaining_courses_sorted = sorted(remaining_courses_with_priority, key=lambda x: x['priority_score'], reverse=True)

    # Use the sorted remaining_courses in dynamically_balance_courses
    remaining_courses_sorted_balanced = dynamically_balance_courses(remaining_courses_sorted)

    # Optionally, you can assign the result back to the remaining_courses variable
    remaining_courses = remaining_courses_sorted_balanced
    
    print(remaining_courses_with_priority)

    completed_gen_elective_1 = count_completed_courses(completed_courses, 'gen_elective_1')
    completed_gen_elective_2 = count_completed_courses(completed_courses, 'gen_elective_2')
    completed_major_elective = count_completed_courses(completed_courses, 'major_elective')

    next_semester = get_next_semester(current_registration_semester)
    itbp480_course = next((course for course in remaining_courses if course['course_id'] == 'ITBP480'), None)
    itbp481_course = next((course for course in remaining_courses if course['course_id'] == 'ITBP481'), None)
    itbp495_course = next((course for course in remaining_courses if course['course_id'] == 'ITBP495'), None)

    # Exclude courses from remaining courses
    remaining_courses = [c for c in remaining_courses if c['course_id'] not in ['ITBP480', 'ITBP481', 'ITBP495']]
    print(remaining_courses)
    year = 2025
    decremented = False

    # Main loop for scheduling courses
    while remaining_courses:
        semester_plan = []
        semester_credit_hours = 0

        # Elective limits check
        if completed_gen_elective_1 >= gen_elective_limit:
            remaining_courses = [c for c in remaining_courses if c['type'] != 'gen_elective_1']
        if completed_gen_elective_2 >= gen_elective_limit:
            remaining_courses = [c for c in remaining_courses if c['type'] != 'gen_elective_2']
        if completed_major_elective >= major_elective_limit:
            remaining_courses = [c for c in remaining_courses if c['type'] != 'major_elective']

        if len(remaining_courses) <= 13 and not decremented: 
            max_credit_hours-= 3
            decremented = True
        
                
        # Fill the semester with courses while respecting max credit hours
        for course in remaining_courses[:]:
            if semester_credit_hours >= max_credit_hours:
                break

            if semester_credit_hours + course['credit_hours'] <= max_credit_hours:
                prerequisites_done = get_prerequisites(course['course_id'], completed_courses)

                if course['type'] == 'mandatory' and prerequisites_done:

                    all_prereqs_done = True
                    coreq_courses_to_add = []
                    
                    corequisites = get_corequisites(course['course_id'])

                    if corequisites:

                        for coreq in corequisites:
                            coreq_course = next((c for c in remaining_courses if c['course_id'] == coreq), None)

                            if coreq_course and coreq_course not in semester_plan:
                                prerequisites_done = get_prerequisites(coreq_course['course_id'], completed_courses)
                                
                                if prerequisites_done:
                                    coreq_courses_to_add.append(coreq_course)  # Store the course temporarily
                                else:
                                    all_prereqs_done = False
                                    break  # If any prerequisite is not met, stop the process
                        
                    # If all prerequisites are done, add the course and its co-requisites
                    if all_prereqs_done:

                        ## add the course 
                        if course in remaining_courses:
                            remaining_courses.remove(course)
                            semester_plan.append(course['course_id'])
                            semester_credit_hours += course['credit_hours']
                        else:
                            print(f"Course {course['course_id']} not found in remaining_courses")

                        ## add its co-courses
                        for coreq_course in coreq_courses_to_add:
                            if coreq_course in remaining_courses: #and semester_credit_hours + coreq_course['credit_hours'] <= (max_credit_hours + 1):
                                semester_plan.append(coreq_course['course_id'])
                                semester_credit_hours += coreq_course['credit_hours']
                                remaining_courses.remove(coreq_course)
                            else:
                                print(f"Course {course['course_id']} not found in remaining_courses")
                        

                elif prerequisites_done and course['type'] != 'mandatory':
                    can_add = False
                    if course['type'] == 'gen_elective_1' and completed_gen_elective_1 < gen_elective_limit:
                        completed_gen_elective_1+=1
                        can_add = True
                    elif course['type'] == 'gen_elective_2' and completed_gen_elective_2 <gen_elective_limit:
                        completed_gen_elective_2+=1
                        can_add = True
                    elif course['type'] == 'major_elective' and completed_major_elective < major_elective_limit:
                        completed_major_elective+=1
                        can_add = True


                    if can_add: 
                        semester_plan.append(course['course_id'])
                        semester_credit_hours += course['credit_hours']
                        remaining_courses.remove(course)



        # Add the plan for this semester to the overall plan
        if semester_plan:
            completed_courses += semester_plan
            plan.append({
                'semester': next_semester + ' ' + str(year),
                'courses': semester_plan
            })
        
        
        # Move to the next semester
        next_semester = 'fall' if next_semester == 'spring' else 'spring'
        year = year + 1 if next_semester == 'spring' else year
    
    # Schedule ITBP courses
    # Schedule ITBP480 in the third last semester
    if itbp480_course:
        plan[-2]['courses'].append(itbp480_course['course_id'])  # Adding to third last semester courses

    # Schedule ITBP481 in the second last semester
    if itbp481_course:
        plan[-1]['courses'].append(itbp481_course['course_id'])  # Adding to second last semester courses

    # Create a new semester for ITBP495 and add it
    if itbp495_course:
        plan.append({
            'semester': 'Last Semester',  # Name it according to your needs
            'courses': [itbp495_course['course_id']]
        })

    print(plan)
    return plan

################################################################################################################################################################

def create_course_graph(student_courses, all_courses, prerequisites, co_courses):
    net = Network(height='500px', width='100%', notebook=True)

    # Create a dictionary for student courses
    student_course_dict = {record['course_id']: record['status'] for record in student_courses}

    # Add all courses to the graph
    for record in all_courses:
        course_id = record['course_id']
        # Determine color based on student course completion and registration status
        if course_id in student_course_dict:
            # Use the status from the student_course_dict
            status = student_course_dict[course_id]
            color = 'green' if status == 'completed' else 'yellow'
        else:
            color = 'lightblue'  # Default color for unregistered courses
        
        net.add_node(course_id, label=record['course_name'], color=color)

    # Add prerequisite edges
    for record in prerequisites:
        course_id = record['course_id']
        prerequisite_id = record['prerequisite_id']
        net.add_edge(course_id, prerequisite_id, label="Prerequisite", color='orange')

    # Add co-course edges
    for record in co_courses:
        course_id = record['course_id']
        co_course_id = record['co_course_id']
        net.add_edge(course_id, co_course_id, label="Co-course", color='purple')

    return net

############################################################################################################################################
            
def generate_transactions(df):    
    for (student_id, term), student_term_data in df.groupby(["Student ID", "Term"]):
        courses = student_term_data["Course Label"].tolist()
    
        if courses:
            yield tuple(sorted(courses)), term

def generate_frequent_itemsets(df, min_support, min_confidence, max_size):
    """
    Generates frequent itemsets without determining the most common term.
    Returns frequent itemsets with support and association rules, only for itemsets whose length is between 2 and max_size.
    """
    transactions = list(generate_transactions(df))
    
    # Extract only the course transactions for Apriori
    course_transactions = [list(trans[0]) for trans in transactions]  # Convert tuples to lists
    
    print('number of transactions')
    print(len(course_transactions))

    # Run Apriori to get frequent itemsets and association rules
    itemsets, rules = ap(course_transactions, min_support=min_support, min_confidence=min_confidence)

    # Dictionary to hold itemsets with their support
    support_dict = {}
    
    print('number of itemsets exceeding the minimum support and confidence')
    print(len(itemsets))

    # Iterate over itemsets dictionary
    for size, items in itemsets.items():
        for itemset, support in items.items():
            # Check if the itemset's length is between 2 and max_size
            if 2 <= len(itemset) <= max_size:
                sorted_tuple = tuple(sorted(itemset))  # Sort the itemset for consistency
                support_dict[sorted_tuple] = support

    # Convert support_dict to DataFrame for easier visualization and analysis
    itemset_df = pd.DataFrame([
        {
            "Itemset": itemset,
            "Support": support
        }
        for itemset, support in support_dict.items()
    ])
    
    print(itemset_df)

    return itemset_df

def calculate_co_occurrence_matrix(df):
    unique_courses = sorted(df["Course Label"].unique())
    course_index = {course: idx for idx, course in enumerate(unique_courses)}
    matrix = np.zeros((len(unique_courses), len(unique_courses)))

    for _, group in df.groupby(["Student ID", "Term"]):
        courses = group["Course Label"].tolist()
        grades = group["Grade Weight"].tolist()

        for (c1, g1), (c2, g2) in combinations(zip(courses, grades), 2):
            idx1, idx2 = course_index[c1], course_index[c2]
            weight = (g1 + g2) / 2  # Average grade weight
            
            matrix[idx1, idx2] += weight
            matrix[idx2, idx1] += weight


    matrix = pd.DataFrame(matrix, index=unique_courses, columns=unique_courses)

    return matrix

def plot_top_pairs(matrix):
    # Flatten the matrix and sort by weight
    pairs = []
    for i in range(len(matrix.columns)):
        for j in range(i + 1, len(matrix.columns)):
            pairs.append((matrix.columns[i], matrix.columns[j], matrix.iloc[i, j]))

    # Sort by co-occurrence weight
    sorted_pairs = sorted(pairs, key=lambda x: x[2], reverse=True)

    # Extract top 10 pairs
    top_pairs = sorted_pairs[:10]
    courses1, courses2, weights = zip(*top_pairs)

    # Only keep course code before '-'
    courses1 = [course.split('-')[0] for course in courses1]
    courses2 = [course.split('-')[0] for course in courses2]

    # Bar plot of top pairs
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_pairs)))  # Better color palette

    bars = plt.barh(range(len(top_pairs)), weights, align='center', color=colors)
    plt.yticks(range(len(top_pairs)), [f"{c1} - {c2}" for c1, c2 in zip(courses1, courses2)], fontsize=12)
    plt.xlabel("Co-occurrence Weight", fontsize=14)
    plt.title("Top 10 Course Pairs by Co-occurrence Weight", fontsize=16)
    
    # Add value annotations to bars
    for bar in bars:
        plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
                 f'{bar.get_width():.2f}', va='center', ha='left', fontsize=12)

    plt.show()

def plot_co_occurrence_matrix(matrix):
    # Create a copy to avoid modifying the original matrix
    new_matrix = matrix.copy()
    
    # Only include the course code before the first hyphen for columns and index
    new_matrix.columns = [course.split('-')[0].strip() for course in new_matrix.columns]
    new_matrix.index = [course.split('-')[0].strip() for course in new_matrix.index]
    
    # Adjust font size based on the longest course code
    max_label_length = max(new_matrix.columns.str.len())  # Find the longest course code
    font_size = max(6, 12 - max_label_length // 15)  # Adjust font size based on label length
    
    # Set the figure size based on the number of columns and rows
    figsize = (len(new_matrix.columns) * 0.5, len(new_matrix.index) * 0.5)
    plt.figure(figsize=figsize)

    # Create the heatmap
    sns.heatmap(new_matrix, cmap='coolwarm', annot=False, fmt=".2f", cbar=True, 
                square=True, linewidths=0.5, linecolor='gray', 
                cbar_kws={'label': 'Co-occurrence Weight'})

    # Title and labels with reduced font size
    plt.title("Course Co-occurrence Matrix", fontsize=10)  # Reduced title font size
    plt.xlabel("Courses", fontsize=8)  # Reduced x-axis label font size
    plt.ylabel("Courses", fontsize=8)  # Reduced y-axis label font size

    # Adjusting tick font size further for smaller labels
    plt.xticks(rotation=90, fontsize=6)  # Smaller font size for x-axis labels
    plt.yticks(rotation=0, fontsize=6)   # Smaller font size for y-axis labels

    plt.show()

def compute_hybrid_scores(df, min_support=0.01, min_confidence=0.5, alpha=0.5, max_size=4):
    co_occurrence_matrix = calculate_co_occurrence_matrix(df)
    support_dict = generate_frequent_itemsets(df, min_support, min_confidence, max_size)

    hybrid_scores = []
    
    # Loop through the support_dict (assuming it's a DataFrame)
    for _, row in support_dict.iterrows():
        itemset = row["Itemset"]
        support = row["Support"]
        
        if len(itemset) >= 2:
            # Calculate average co-occurrence for the itemset
            co_occurrence_sum = 0
            count = 0
            for i, c1 in enumerate(itemset):
                for c2 in itemset[i+1:]:  # Iterate over pairs within the itemset
                    co_occurrence_sum += co_occurrence_matrix.loc[c1, c2]
                    count += 1
            
            # Calculate the average co-occurrence if count is greater than 0
            average_co_occurrence = co_occurrence_sum / count if count > 0 else 0
            
            # Compute hybrid score
            hybrid_score = alpha * support + (1 - alpha) * average_co_occurrence
            hybrid_scores.append({"Itemset": itemset, "Support": support, "Average Co-occurrence": average_co_occurrence, "Hybrid Score": hybrid_score})

    return pd.DataFrame(hybrid_scores).sort_values("Hybrid Score", ascending=False)

def build_student_vector(course_list, grade_list, term_list, all_courses, target_terms, ind):    
    vector = []
        
    for course in all_courses:
        if course in course_list:
            index = course_list.index(course)
                        
            grade_value = grade_weights.get(grade_list[index], 0) / 4.0  # Normalize grade to [0,1]
            
            # Compute term similarity as an inverse function of term difference
            term_difference = abs(term_list[index] - target_terms.get(course, term_list[index]))  # Default to own term if missing
            term_weight = 1 / (1 + term_difference)  # Higher weight for closer terms
            
            # Final value incorporates both grade and term weight
            vector.append(grade_value * term_weight)
        else:
            vector.append(0)
    return vector

def recommend_courses(student_data, target_student, k, is_new_student):
    """Recommend courses based on similar students using cosine similarity, factoring in term similarity."""
    
    all_courses = sorted(set(course.split(' -')[0] for courses in student_data["Course Label"] for course in courses))
    print(all_courses)
    
    student_data["Cleaned Course Label"] = student_data["Course Label"].apply(
    lambda courses: [course.split(' -')[0] for course in courses])
    
    target_terms = {course: term for course, term in zip(target_student["Completed Courses"], target_student["Terms"])}

    target_vector = build_student_vector(
        target_student["Completed Courses"], target_student["Grades"], target_student["Terms"], all_courses, target_terms, 1
    )
    

    student_data["Vectors"] = student_data.apply(
    lambda row: build_student_vector(row["Cleaned Course Label"], row["Grade"], row["Term"], all_courses, target_terms, 2),
    axis=1
    )   

    # Compute cosine similarity
    student_data["Similarity"] = cosine_similarity([target_vector], list(student_data["Vectors"]))[0]

    similar_students = student_data if is_new_student else student_data.nlargest(k, "Similarity")

    # Collect course terms from similar students
    course_terms = defaultdict(list)
    for _, row in similar_students.iterrows():
        for course, term in zip(row["Course Label"], row["Term"]):
            course_terms[course].append(term)
            
    # Compute average term for each course
    avg_course_terms = {
        course: (math.floor(np.mean(terms)) if np.mean(terms) % 1 == 0.5 else round(np.mean(terms)))
        for course, terms in course_terms.items()
    }

    recommended_courses = sorted(
        [(course, avg_term) for course, avg_term in avg_course_terms.items()
         if course not in target_student["Completed Courses"]],
        key=lambda x: x[1]
    )

    return recommended_courses, similar_students

def get_target_student_from_graph(graph, student_id):
    query = """
    MATCH (s:Student {id: $student_id})-[r:completed]->(c:Course)
    RETURN c.id AS course_id, r.grade AS grade, r.term_sequence AS term_sequence
    ORDER BY r.term_sequence, course_id
    """
    results = graph.run(query, student_id=student_id).data()
    
    completed_courses = [row['course_id'] for row in results]
    grades = [row['grade'] for row in results]
    terms = [row['term_sequence'] for row in results]
    
    return {
        "Completed Courses": completed_courses,
        "Grades": grades,
        "Terms": terms,
    }
    
def prepare_prompt_second_approach(student_id, student_major, is_new_student):
    
    # Read student course data
    if student_major == 'CS': 
        file_path = "data/CS_Student_Courses.xlsx"
        required_courses["major_elective"] = 4
    elif student_major == 'IS': 
        file_path = "data/IS_Student_Courses.xlsx"
        required_courses["major_elective"] = 3
    elif student_major == 'IT': 
        file_path = "data/IT_Student_Courses.xlsx"
        required_courses["major_elective"] = 2

    data = pd.read_excel(file_path)

    # Process course data
    data["Grade Weight"] = data["Grade"].map(grade_weights)
    data["Course Label"] = data["Subject Code"] + " - " + data["Course Title"]
    data["Term"] = data.groupby("Student ID")["Term"].rank(method="dense").astype(int)

    student_data = (
        data.groupby("Student ID")
        .agg({"Course Label": list, "Grade": list, "Term": list})
        .reset_index()
    )
    
    target_student = get_target_student_from_graph(graph, student_id)

    # Prepare course completion plan
    completed_courses = target_student['Completed Courses']
    courses_list = get_all_courses_for_major(student_major)


    prerequisites_dict = get_prerequisites_dict()
    cocourses_dict = get_cocourses_dict()

    for course in courses_list:
        if course["course_id"] == "CSBP320" and student_major == 'IS':
            course["type"] = "mandatory"
            
    # Extract only course_id values
    course_ids = {course["course_id"] for course in courses_list}  # Using a set for faster lookup

    # Filter the DataFrame
    data = data[data["Subject Code"].isin(course_ids)]

    # Generate recommendations
    recommended_courses, similar_students = recommend_courses(student_data, target_student, k=15, is_new_student=is_new_student)

    hybrid_scores_df = compute_hybrid_scores(data, min_support=0.02, min_confidence= 0.5, alpha=0.8, max_size=5)

    course_recommendations = defaultdict(set)

    # Iterate over the hybrid scores DataFrame
    for _, row in hybrid_scores_df.iterrows():
        courses = row["Itemset"]  # This is now a tuple of multiple courses
        for course in courses:  # Iterate through each course in the tuple
            course_recommendations[course].update(set(courses) - {course})

    course_recommendations = {
        course.split(" - ")[0]: [c.split(" - ")[0] for c in recommendations]
        for course, recommendations in course_recommendations.items()
    }

    # course embeddings from KG 
    course_embeddings = get_course_embeddings_for_major(graph, student_major)
    course_ids = [course_id for course_id, _, _ in course_embeddings]
    course_vectors = np.array([embedding for _, _, embedding in course_embeddings])
    course_similarity_matrix = cosine_similarity(course_vectors)

    threshold = 0.7
    recommended_courses_embd = {}

    for i, course_id in enumerate(course_ids):
        if not course_id.startswith("topic_"):
            recommended_courses_embd[course_id] = [
                course_ids[j]
                for j, similarity in enumerate(course_similarity_matrix[i])
                if similarity >= threshold and i != j and not course_ids[j].startswith("topic_")
            ]

    # Filter out empty recommendations
    filtered_courses = {k: v for k, v in recommended_courses_embd.items() if v}


    # Format output more concisely
    similar_courses_text = "\n".join(
        f"{course_id} → {', '.join(courses)}" for course_id, courses in filtered_courses.items()
    )

    all_courses_with_details = [
        {
            "code": course["course_id"],
            "name": course["name"],
            "credit_hours": course["credit_hours"],
            "type": course["type"],
            "prerequisites": prerequisites_dict.get(course["course_id"], []),
            "corequisites": cocourses_dict.get(course["course_id"], []),
            "term": (
            next((term for c, term in recommended_courses if c.split(" - ")[0] == "CSBP219"), "N/A")
            if course["course_id"] == "CSBP221"
            else next((term for c, term in recommended_courses if c.split(" - ")[0] == "CSBP119"), "N/A")
            if course["course_id"] == "CSBP121"
            else next((term for c, term in recommended_courses if c.split(" - ")[0] == course["course_id"]), "N/A")
            ),
            "frequent_courses": course_recommendations.get(course["course_id"]) , 
            "similar_courses":[sim_course for sim_course in filtered_courses.get(course["course_id"], []) if sim_course not in course_recommendations.get(course["course_id"], [])] or ["N/A"]
        }
        for course in courses_list
    ]


    # Student profile
    student_data = {
        "completed_courses": completed_courses,
        "gpa": 3.5,
    }

    sorted_courses = sorted(
        (course for course in all_courses_with_details 
        if course['code'] not in student_data['completed_courses']),
        key=lambda x: float('inf') if x['term'] == "N/A" else x['term']  # Push "N/A" terms to the end
    )


    # Step 1: Count completed courses per category
    completed_counts = {
        "mandatory": sum(1 for course in all_courses_with_details if course["code"] in student_data["completed_courses"] and course["type"] == "mandatory"),
        "major_elective": sum(1 for course in all_courses_with_details if course["code"] in student_data["completed_courses"] and course["type"] == "major_elective"),
        "gen_elective_1": sum(1 for course in all_courses_with_details if course["code"] in student_data["completed_courses"] and course["type"] == "gen_elective_1"),
        "gen_elective_2": sum(1 for course in all_courses_with_details if course["code"] in student_data["completed_courses"] and course["type"] == "gen_elective_2"),
    }

    # Step 2: Calculate remaining courses per category
    remaining_courses = {
        category: required_courses[category] - completed_counts[category]
        for category in required_courses
    }

    # Step 3: Dynamically generate course lists (only if remaining > 0)
    remaining_course_sections = []
    if remaining_courses["mandatory"] > 0:
        remaining_course_sections.append(f"### **Mandatory Courses (Select all of them)**\n" + "\n".join(
            f"- {course['code']} ({course['name']}, {course['credit_hours']} credit hours, 3 contact hours), "
            f"Prerequisites: {', '.join(course['prerequisites']) if course['prerequisites'] else 'None'}, "
            f"Corequisites: {', '.join(course['corequisites']) if course['corequisites'] else 'None'}, "
            f"Mostly taken with: {', '.join(course['frequent_courses']) if course['frequent_courses'] else 'No Enough Information'}, "
            f"Similar Courses: {', '.join(course['similar_courses']) if course['similar_courses'] else 'No Enough Information'}, "
            f"Suggested Term: {course['term']}"
            for course in sorted_courses
            if course['code'] not in student_data['completed_courses'] and course['type'] == 'mandatory'
        ))

    if remaining_courses["major_elective"] > 0:
        remaining_course_sections.append(f"### **Select EXACTLY {remaining_courses['major_elective']} from Major Electives for the entire plan**\n" + "\n".join(
            f"- {course['code']} ({course['name']}, {course['credit_hours']} credit hours, 3 contact hours), "
            f"Prerequisites: {', '.join(course['prerequisites']) if course['prerequisites'] else 'None'}, "
            f"Corequisites: {', '.join(course['corequisites']) if course['corequisites'] else 'None'}, "
            f"Mostly taken with: {', '.join(course['frequent_courses']) if course['frequent_courses'] else 'No Enough Information'}, "
                    f"Similar Courses: {', '.join(course['similar_courses']) if course['similar_courses'] else 'No Enough Information'}, "
            f"Suggested Term: {course['term']}"
            for course in sorted_courses
            if course['code'] not in student_data['completed_courses'] and course['type'] == 'major_elective'
        ))

    if remaining_courses["gen_elective_1"] > 0:
        remaining_course_sections.append(f"### **Select EXACTLY {remaining_courses['gen_elective_1']} from General Elective 1 for the entire plan**\n" + "\n".join(
            f"- {course['code']} ({course['name']}, {course['credit_hours']} credit hours, 3 contact hours), "
            f"Prerequisites: {', '.join(course['prerequisites']) if course['prerequisites'] else 'None'}, "
            f"Corequisites: {', '.join(course['corequisites']) if course['corequisites'] else 'None'}, "
            f"Mostly taken with: {', '.join(course['frequent_courses']) if course['frequent_courses'] else 'No Enough Information'}, "
            f"Similar Courses: {', '.join(course['similar_courses']) if course['similar_courses'] else 'No Enough Information'}, "
            f"Suggested Term: {course['term']}"
            for course in sorted_courses
            if course['code'] not in student_data['completed_courses'] and course['type'] == 'gen_elective_1'
        ))

    if remaining_courses["gen_elective_2"] > 0:
        remaining_course_sections.append(f"### **Select EXACTLY {remaining_courses['gen_elective_2']} from General Elective 2 for the entire plan**\n" + "\n".join(
            f"- {course['code']} ({course['name']}, {course['credit_hours']} credit hours, 3 contact hours), "
            f"Prerequisites: {', '.join(course['prerequisites']) if course['prerequisites'] else 'None'}, "
            f"Corequisites: {', '.join(course['corequisites']) if course['corequisites'] else 'None'}, "
            f"Mostly taken with: {', '.join(course['frequent_courses']) if course['frequent_courses'] else 'No Enough Information'}, "
            f"Similar Courses: {', '.join(course['similar_courses']) if course['similar_courses'] else 'No Enough Information'}, "
            f"Suggested Term: {course['term']}"
            for course in sorted_courses
            if course['code'] not in student_data['completed_courses'] and course['type'] == 'gen_elective_2'
        ))

    # Step 4: Insert dynamically generated course list into the prompt
    course_list_section = "\n\n".join(remaining_course_sections)

    # Update the prompt
    prompt = f"""
    You are a university course advisor. Generate a full course 130 hours plan based on the student's progress.

    ## **Student Data**
    - **Completed Courses:** {', '.join(student_data['completed_courses'])}
    - The student has already completed {sum(course['credit_hours'] for course in all_courses_with_details if course['code'] in student_data['completed_courses'])}  credit hours.
    - The remaining required credit hours = {130 - sum(course['credit_hours'] for course in all_courses_with_details if course['code'] in student_data['completed_courses'])}.
    - **GPA:** {student_data['gpa']}
    - **Current Term Sequence:** {max(target_student["Terms"]) + 1 if target_student["Terms"] else 1}

    ---

    ## **Available Courses (Grouped by Category, Excluding Completed Courses)**

    {course_list_section} 

    ---

    - For most semesters, the total credit hours should be between 15 and 16 credit hours.
    - Each semester must have no less than 4 courses and no less than 12 credit hours.
    - The overall average should stay around 15 to 16 credit hours per semester to maintain a balanced workload.
    - The overall average should stay around 15 to 18 contact hours per semester to maintain a balanced workload.
    - A semester cannot have more than 18 credit hour.
    - A semester should have a maximum of 18 contact hours. 
    - Prioritize courses that are prerequisites to many other courses. 
    - A student with 0 completed courses must finish the degree in 9 semesters. 
    - Aim to have the plan finished in 8 semesters (or less) including the completed semesters. For example, if the student completed 1 semester, then aim to finish the plan in 8 semesters. 
    - Do not list a course in a semester unless all its prerequisite courses have already been completed in previous semesters.
    - A course and its prerequisites cannot be listed in the same semester. 
    - two courses that are corequisites must be listed together in the same semester only and cannot be separated.
    - Total Credit Hours Must Equal 130 credit hours. 
    - The Internship course (ITBP495) must be taken alone in the final semester, with no other courses.
    - ITBP480 and ITBP481 must be taken in the two semesters right before the internship semester.
    - ITBP481 must be taken immediately before the internship semester.
    - You cannot change the type of a course.

    ---

    ## **Output Format**
    Provide the recommended courses for each term in the following format:
        {{
            'Spring YEAR': [
                {{'id': 'course_code', 'name': 'course_name', 'credit_hours': credit_hours}},
                ...
            ],
            'Fall YEAR': [
                {{'id': 'course_code', 'name': 'course_name', 'credit_hours': credit_hours}},
                ...
            ],
            ...
        }}
        Where `YEAR` starts from 2025. List all terms and courses until graduation in this format. 
        Replace `course_code`, `course_name`, and `credit_hours` with the actual course details. Use json format for the output with double quotes. 
        Only return the dictionary as the output, without any additional explanations or comments.
    """
    
    return prompt

def generate_plan_second_approach(prompt):
    plan = None
    try:
        completion = client.chat.completions.create(
            model = 'anthropic/claude-3.5-sonnet',
            messages=[
                {"role": "system", "content": "You are a helpful academic advisor. Respond with only the requested structured format."},
                {"role": "user", "content": prompt},
            ]
        )
        
        print(completion)
        print(completion.choices[0].message.content)
        plan = completion.choices[0].message.content
    except Exception as e:
        print("Error during API call:", e)
    
    return plan