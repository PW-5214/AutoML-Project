import streamlit as st
import pandas as pd
import os
import sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.insert(0,str(Path(__file__).parent))

from src.pipeline import MLpipeline


def _safe_dataframe_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    safe_df = df.copy()
    for col in safe_df.columns:
        safe_df[col] = safe_df[col].map(
            lambda value: value
            if isinstance(value, (str, int, float, bool, type(None)))
            else str(value)
        )
        if safe_df[col].dtype == "object":
            safe_df[col] = safe_df[col].astype(str)
    return safe_df

st.set_page_config(page_title="AutoML")

st.title("AutoML Platform")

with st.sidebar:
    st.header("Configuration")
    task = st.selectbox("Task Type",['Classification',"Regression"])

    file = st.file_uploader("Upload CSV",type=['csv','xlsx'])

    cols = []
    if file:
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        cols = df.columns.to_list()
        st.success(f"✅ {df.shape[0]} rows, {df.shape[1]} cols")
    
    target = st.selectbox("Target Column",cols if cols else ["Select file first"])
    run = st.button("Run Pipeline",use_container_width=True)

if file and target != "Select file first":
    if run:
        os.makedirs("data/raw",exist_ok=True)
        path = f"data/raw/uploaded.csv"

        with open(path,"wb") as f:
            f.write(file.getbuffer())
        
        with st.spinner("Running Pipeline..."):
            try:
                pipeline = MLpipeline()
                results  = pipeline.run_pipeline(path,target,task=task.lower())

                st.session_state.results = results
                st.session_state.leaderboard = pipeline.leaderboard
                st.success("✅ Pipeline completed!")
            
            except Exception as e:
                st.error(f"Error:{str(e)}")
    
    if 'results' in st.session_state:
        results = st.session_state.results

        #Top metrics
        st.markdown("📊 Top Results")
        col1,col2,col3 = st.columns(3)

        with col1:
            st.metric("Best Model",results['model_name'])
        
        with col2:
            evals = results['evaluation']
            score = evals.get('accuracy') or evals.get('R2_score',0)
            st.metric("Score",f"{score:.4f}")
        
        with col3:
            st.metric("Total Models",len(st.session_state.leaderboard.leaderboard_data))

        st.markdown('---')

        ## Tabs
        tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(["📈 Leaderboard", "📋 Metrics", "📄 Report", "📥 Download","📊 Visualizations","📊 EDA"])

        with tab1:
            st.subheader("Model leaderboard")
            lb = st.session_state.leaderboard.get_leaderboard()
            st.dataframe(_safe_dataframe_for_streamlit(lb),use_container_width=True)

        with tab2:
            st.subheader("Best Model Metrics")
            evals = results['evaluation']

            for metric,value in evals.items():
                if isinstance(value,(int,float)):
                    st.metric(metric,f"{value:.4f}")
            
            st.markdown("## All metrics")
            metrics_text = "\n".join([f"• **{k}:** {v}" for k, v in evals.items()])
            st.markdown(metrics_text)
        
        with tab3:
            st.subheader("Full Report")

            try:
                with open(results['report_path'],'r') as f:
                    report = f.read()

                st.text_area("Report Content",report,height=500,disabled=True)
            except:
                st.warning("Report Not available")
        
        with tab4:
            st.subheader("Download File")

            col1,col2,col3 = st.columns(3)

            with col1:
                try:
                    with open(results['report_path'],'r') as f:
                        st.download_button("" \
                        "Download Report",
                        f.read(),
                        "report.txt",
                        use_container_width=True
                        )
                except:
                    st.write("Report not available")
            
            with col2:
                lb = st.session_state.leaderboard.get_leaderboard()
                st.download_button(
                    "Download Leaderboard",
                    lb.to_csv(index=False),
                    "leaderboard.csv",
                    use_container_width=True
                )
            
            with col3:
                try:
                    with open(results['model_path'],'rb') as f:
                        st.download_button(
                            "Download Model",
                            f.read(),
                            f"{results['model_name']}.pkl",
                            use_container_width=True
                        )
                except:
                    st.write("Model not Available")

        with tab5:
            st.subheader("Model Performance Visualizations")

            lb = st.session_state.leaderboard.get_leaderboard()
            metric_cols = [col for col in lb.columns if col not in ['Rank','Model']]

            if metric_cols:
                col1,col2 = st.columns(2)

                with col1:
                    fig1 = px.bar(
                        lb,
                        x='Model',
                        y=metric_cols[-1],
                        color = metric_cols[-1],
                        color_continuous_scale = 'Viridis',
                        title=f"Model Performance ({metric_cols[-1]})"

                    )
                    fig1.update_layout(height=400,xaxis_tickangle=-45)
                    st.plotly_chart(fig1,use_container_width=True)
                
                with col2:
                    fig2 = px.box(
                        lb,
                        y=metric_cols[-1],
                        title=f"Score Distribution"
                    )
                    fig2.update_layout(height=400)
                    st.plotly_chart(fig2,use_container_width=True)

            col1,col2 = st.columns(2)

            with col1:
                fig3 = px.pie(
                    values =[1]*len(lb),
                    names=lb['Model'],
                    title="Models Count"
                )
                st.plotly_chart(fig3,use_container_width=True)
            
            with col2:
                if metric_cols:
                    stats_data = {
                        'Category':['Best','Average','Worst'],
                        'Score':[
                            lb[metric_cols[-1]].max(),
                            lb[metric_cols[-1]].mean(),
                            lb[metric_cols[-1]].min()
                        ]
                    }

                    fig4 = px.bar(
                        stats_data,
                        x='Category',
                        y='Score',
                        color='Score',
                        color_continuous_scale ='Blues',
                        title ='Performance Summary',
                        text = 'Score'
                    )
                    fig4.update_layout(height=400)
                    st.plotly_chart(fig4,use_container_width=True)
            
            if results['feature_importance'] is not None:
                st.markdown('---')
                st.subheader('Top Features')

                feat_imp = results['feature_importance'].head(10)
                fig5 = px.bar(
                    feat_imp,
                    x='importance',
                    y='feature',
                    orientation ='h',
                    color_continuous_scale ='Greens',
                    title ='Top 10 important Features'
                )
                fig5.update_layout(height=400)
                st.plotly_chart(fig5,use_container_width=True)
        
        with tab6:
            st.subheader("📊 Exploratory Data Analysis")
            
            col1,col2,col3 = st.columns(3)
            
            with col1:
                st.markdown("### Dataset Info")
                st.write(f"**Rows:** {df.shape[0]}")
                st.write(f"**Columns:** {df.shape[1]}")
                st.write(f"**Memory:** {df.memory_usage().sum() / 1024:.2f} KB")
            
            with col2:
                st.markdown("### Data Types")
                types = df.dtypes.value_counts()
                for dtype, count in types.items():
                    st.write(f"**{dtype}:** {count}")
            
            with col3:
                st.markdown("### Missing Values")
                missing = df.isnull().sum()
                if missing.sum() == 0:
                    st.write("✅ No missing values")
                else:
                    for col_name, count in missing[missing > 0].items():
                        st.write(f"**{col_name}:** {count}")
            
            st.markdown("---")
            
            st.markdown("### 📈 Correlation Matrix")
            numeric_df = df.select_dtypes(include=['number'])
            
            if len(numeric_df.columns) > 1:
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax, cbar=True)
                st.pyplot(fig)
            else:
                st.info("Not enough numeric columns for correlation matrix")
            
            st.markdown("---")
            
            st.markdown("### 📊 Distribution Plots")
            
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if numeric_cols:
                cols_to_plot = numeric_cols[:6]
                
                for i, col in enumerate(cols_to_plot):
                    if i % 2 == 0:
                        col1, col2 = st.columns(2)
                    
                    with (col1 if i % 2 == 0 else col2):
                        fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}")
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, width='stretch')
            
            st.markdown("---")
            
            st.markdown("### 🎯 Target Variable Distribution")
            
            fig = px.bar(
                df[target].value_counts(),
                title=f"Distribution of {target}",
                labels={'index': target, 'value': 'Count'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')


else:
    st.info("Step 1:** Upload CSV file")
    st.info("Step 2:** Select target column")
    st.info("Step 3:** Click  Run Pipeline")

