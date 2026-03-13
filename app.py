import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import random
import os

# 페이지 설정
st.set_page_config(page_title="Online Retail 분석 및 추천 대시보드", layout="wide")

# --- 데이터 로딩 및 전처리 ---
@st.cache_data
def load_data():
    # 데이터 경로를 더 유연하게 탐색 (로컬 및 배포 환경 대응)
    possible_paths = [
        "data/online_retail.parquet",
        "online-retail/data/online_retail.parquet",
        os.path.join(os.path.dirname(__file__), "data/online_retail.parquet"),
        os.path.join(os.path.dirname(__file__), "online-retail/data/online_retail.parquet")
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
            
    if data_path is None:
        st.error("데이터 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        st.stop()
        
    df = pd.read_parquet(data_path)
    
    # 데이터 품질 처리
    df = df.dropna(subset=['CustomerID', 'Description'])
    df['CustomerID'] = df['CustomerID'].astype(int)
    
    # 취소 주문 처리 (Invoice가 C로 시작하는 경우 제외)
    df = df[~df['InvoiceNo'].str.startswith('C')]
    
    # 음수 수량 및 가격 제거
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    
    # 시간 데이터 처리
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Year'] = df['InvoiceDate'].dt.year
    df['Month'] = df['InvoiceDate'].dt.month
    df['MonthYear'] = df['InvoiceDate'].dt.to_period('M').astype(str)
    df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
    
    # 매출 데이터 생성
    df['Revenue'] = df['Quantity'] * df['UnitPrice']
    
    return df

# --- 분석 함수 ---
@st.cache_data
def get_eda_stats(df):
    stats = {
        "rows": len(df),
        "customers": df['CustomerID'].nunique(),
        "products": df['StockCode'].nunique(),
        "invoices": df['InvoiceNo'].nunique(),
        "total_revenue": df['Revenue'].sum(),
        "start_date": df['InvoiceDate'].min().strftime('%Y-%m-%d'),
        "end_date": df['InvoiceDate'].max().strftime('%Y-%m-%d')
    }
    return stats

# --- 추천 엔진 ---
@st.cache_resource
def build_recommendation_models(df):
    # 1. 콘텐츠 기반 추천 준비
    product_info = df[['StockCode', 'Description']].drop_duplicates(subset=['StockCode']).set_index('StockCode')
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(product_info['Description'])
    
    # 2. 협업 필터링 준비
    user_item_matrix = df.pivot_table(index='CustomerID', columns='StockCode', values='Quantity', aggfunc='sum').fillna(0)
    user_item_matrix = user_item_matrix.applymap(lambda x: 1 if x > 0 else 0)
    user_similarity = cosine_similarity(user_item_matrix)
    user_sim_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
    
    return product_info, tfidf, tfidf_matrix, user_item_matrix, user_sim_df

def get_content_recommendations(user_id, df, product_info, tfidf, tfidf_matrix, n=10):
    user_history = df[df['CustomerID'] == user_id]['StockCode'].unique()
    if len(user_history) == 0: return None
    
    # 유저 프로필 생성 (사용자가 산 상품들의 TF-IDF 평균)
    user_item_indices = [product_info.index.get_loc(sc) for sc in user_history if sc in product_info.index]
    if not user_item_indices: return None
    
    user_profile = tfidf_matrix[user_item_indices].mean(axis=0)
    user_profile = np.asarray(user_profile)
    
    sim_scores = cosine_similarity(user_profile, tfidf_matrix).flatten()
    sim_idx = sim_scores.argsort()[::-1]
    
    # 이미 구매한 상품 제외
    recommended_idx = [i for i in sim_idx if product_info.index[i] not in user_history][:n]
    
    res = []
    for i in recommended_idx:
        res.append({
            "Rank": len(res)+1,
            "StockCode": product_info.index[i],
            "Description": product_info.iloc[i]['Description'],
            "Similarity Score": f"{sim_scores[i]:.4f}"
        })
    return pd.DataFrame(res)

def get_cf_recommendations(user_id, user_item_matrix, user_sim_df, product_info, n=10):
    if user_id not in user_sim_df.index: return None
    
    similar_users = user_sim_df[user_id].sort_values(ascending=False)[1:11]
    similar_users_idx = similar_users.index
    
    similar_users_items = user_item_matrix.loc[similar_users_idx].T.dot(similar_users.values)
    
    user_bought = user_item_matrix.loc[user_id]
    user_bought = user_bought[user_bought > 0].index
    
    recommendations = similar_users_items.drop(user_bought).sort_values(ascending=False).head(n)
    
    res = []
    for sc, score in recommendations.items():
        desc = product_info.loc[sc, 'Description'] if sc in product_info.index else "Unknown"
        res.append({
            "Rank": len(res)+1,
            "StockCode": sc,
            "Description": desc,
            "Score": f"{score:.4f}"
        })
    return pd.DataFrame(res)

# --- 메인 앱 ---
def main():
    df = load_data()
    stats = get_eda_stats(df)
    
    st.sidebar.title("Online Retail 분석실")
    menu = st.sidebar.radio("메뉴 선택", ["대시보드 (EDA)", "개인화 추천 시스템"])
    
    if menu == "대시보드 (EDA)":
        st.title("🛍️ Online Retail 데이터 분석 대시보드")
        
        # KPI Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("총 매출", f"£{stats['total_revenue']:,.0f}")
        m2.metric("총 고객 수", f"{stats['customers']:,}명")
        m3.metric("총 상품 수", f"{stats['products']:,}개")
        m4.metric("총 주문 수", f"{stats['invoices']:,}건")
        
        st.divider()
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("🌍 국가별 매출 TOP 10")
            country_rev = df.groupby('Country')['Revenue'].sum().sort_values(ascending=False).head(10).reset_index()
            fig = px.bar(country_rev, x='Revenue', y='Country', orientation='h', color='Revenue', color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.subheader("📅 월별 매출 추이")
            monthly_rev = df.groupby('MonthYear')['Revenue'].sum().reset_index()
            fig = px.line(monthly_rev, x='MonthYear', y='Revenue', markers=True)
            st.plotly_chart(fig, use_container_width=True)
            
        c3, c4 = st.columns(2)
        
        with c3:
            st.subheader("📅 요일별 주문 건수")
            order_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Sunday']
            day_counts = df.groupby('DayOfWeek')['InvoiceNo'].nunique().reindex(order_days).reset_index()
            fig = px.bar(day_counts, x='DayOfWeek', y='InvoiceNo', color='InvoiceNo')
            st.plotly_chart(fig, use_container_width=True)
            
        with c4:
            st.subheader("🔥 인기 키워드 (TF-IDF)")
            text_data = df['Description'].unique()
            vectorizer = TfidfVectorizer(stop_words='english', max_features=20)
            tfidf_matrix = vectorizer.fit_transform(text_data)
            scores = tfidf_matrix.sum(axis=0).A1
            keywords = pd.DataFrame({'Keyword': vectorizer.get_feature_names_out(), 'Score': scores}).sort_values('Score', ascending=False)
            fig = px.bar(keywords, x='Score', y='Keyword', orientation='h')
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.title("🎯 개인별 맞춤 추천 시스템")
        
        product_info, tfidf, tfidf_matrix, user_item_matrix, user_sim_df = build_recommendation_models(df)
        
        if 'random_users' not in st.session_state:
            st.session_state.random_users = random.sample(user_item_matrix.index.tolist(), 30)
            
        if st.sidebar.button("랜덤 사용자 다시 뽑기"):
            st.session_state.random_users = random.sample(user_item_matrix.index.tolist(), 30)
            
        st.subheader("👥 추천 대상 사용자 선택 (랜덤 30명)")
        selected_user = st.selectbox("사용자를 선택하세요", st.session_state.random_users)
        
        if selected_user:
            user_df = df[df['CustomerID'] == selected_user]
            
            u1, u2, u3 = st.columns(3)
            with u1:
                st.info(f"**고객 ID:** {selected_user}")
            with u2:
                st.success(f"**총 구매 금액:** £{user_df['Revenue'].sum():,.2f}")
            with u3:
                st.warning(f"**구매 상품수:** {user_df['StockCode'].nunique()}개")
            
            st.divider()
            
            # 추천 탭
            tab1, tab2 = st.tabs(["💡 추천 결과 비교", "📜 구매 이력"])
            
            with tab1:
                st.markdown("### 하이브리드 추천 엔진 결과")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("#### 1. 콘텐츠 기반 추천 (TF-IDF)")
                    st.caption("사용자가 구매한 상품의 설명을 분석하여 유사한 상품을 추천합니다.")
                    c_recs = get_content_recommendations(selected_user, df, product_info, tfidf, tfidf_matrix)
                    if c_recs is not None:
                        st.table(c_recs)
                    else:
                        st.write("데이터가 부족하여 추천할 수 없습니다.")
                        
                with col2:
                    st.write("#### 2. 협업 필터링 추천 (User-based)")
                    st.caption("비슷한 구매 성향을 가진 다른 사용자의 데이터를 기반으로 추천합니다.")
                    cf_recs = get_cf_recommendations(selected_user, user_item_matrix, user_sim_df, product_info)
                    if cf_recs is not None:
                        st.table(cf_recs)
                    else:
                        st.write("데이터가 부족하여 추천할 수 없습니다.")
                        
            with tab2:
                st.write("#### 최근 구매 상품 TOP 10")
                top_bought = user_df.groupby(['StockCode', 'Description'])['Quantity'].sum().sort_values(ascending=False).head(10).reset_index()
                st.dataframe(top_bought, use_container_width=True)

if __name__ == "__main__":
    main()
