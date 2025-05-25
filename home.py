import streamlit as st
import pandas as pd
import plotly.express as px
import helper

st.set_page_config(page_title="Student Dashboard", layout="wide")

# Simulated logged-in user
student_id = '201800871'
student_major = helper.get_major(student_id)
completed_data = helper.get_target_student_from_graph(helper.graph, student_id)
completed_courses = completed_data["Completed Courses"]
grades = completed_data["Grades"]
terms = completed_data["Terms"]
registered_courses = helper.get_registered_courses(student_id)

# Title and Summary
st.markdown("<h1 style='color:#4A90E2;'>📊 Student Dashboard</h1>", unsafe_allow_html=True)
st.markdown(f"<b style='font-size:16px;'>Student ID:</b> <code>{student_id}</code>", unsafe_allow_html=True)
st.markdown(f"<b style='font-size:16px;'>Major:</b> <code>{student_major}</code>", unsafe_allow_html=True)

# ---- Summary Cards ----
total_completed = len(completed_courses)
total_registered = len(registered_courses)

st.markdown("### 📌 Summary")
col1, col2, col3 = st.columns(3)
col1.metric("✅ Completed Courses", total_completed)
col2.metric("📝 Registered Courses", total_registered)
col3.metric("📚 Courses This Term", len(registered_courses))

# ---- Graphs Section ----
st.markdown("### 🎯 Academic Performance Visualizations")

if completed_courses:

    df = pd.DataFrame({
        "Course": completed_courses,
        "Grade": grades,
        "Term": terms
    })

    # 2-column layout
    col1, col2 = st.columns(2)

    # Graph 1: Grade distribution
    with col1:
        fig1 = px.histogram(
            df, x="Grade", title="Grade Distribution", color_discrete_sequence=["#009688"]
        )
        fig1.update_layout(title_font_size=16, xaxis_title="Grade", yaxis_title="Count")
        st.plotly_chart(fig1, use_container_width=True)

    # Graph 2: Courses per term
    with col2:
        term_counts = df.groupby("Term")["Course"].count().reset_index()
        fig2 = px.bar(
            term_counts, x="Term", y="Course", title="Courses per Term", color_discrete_sequence=["#3f51b5"]
        )
        fig2.update_layout(title_font_size=16, yaxis_title="Course Count")
        st.plotly_chart(fig2, use_container_width=True)


# ---- Completed & Registered Courses Tables ----
st.markdown("### 📚 Completed & Registered Courses")

if completed_courses:
    completed_df = pd.DataFrame({
        "Course ID": completed_courses,
        "Grade": grades,
        "Term": terms
    })
    st.subheader("✅ Completed Courses")
    st.dataframe(completed_df.style.set_properties(**{'font-size': '14px'}), use_container_width=True)

if registered_courses:
    registered_df = pd.DataFrame({
        "Course ID": registered_courses,
        "Course Name": [helper.get_course_details(cid)['name'] for cid in registered_courses]
    })
    st.subheader("📝 Registered Courses")
    st.dataframe(registered_df.style.set_properties(**{'font-size': '14px'}), use_container_width=True)


# ---- Knowledge Graph Visualization ----
student_courses, all_courses, prerequisites, co_courses = helper.get_courses_and_relationships(
    helper.graph, student_id, student_major
)

if all_courses:
    st.markdown("### Course Graph")

    course_graph = helper.create_course_graph(student_courses, all_courses, prerequisites, co_courses)

    path = '/tmp/course_graph.html'
    course_graph.show(path)
    with open(path, 'r', encoding='utf-8') as HtmlFile:
        graph_html = HtmlFile.read()

    st.components.v1.html(graph_html, height=600, scrolling=True)
