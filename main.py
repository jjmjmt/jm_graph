import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# 1. 브라우저 탭 설정 및 앱 제목 구성 (아이콘 지정)
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 영화 유형 나누기 앱")

# 2. 데이터 불러오기 경로 설정 (로컬 파일 및 온라인 예비 경로 지정)
LOCAL_FILE = "kobis_movies.csv"
REMOTE_URL = "https://githubusercontent.com"

@st.cache_data
def load_data():
    # 로컬 폴더에 파일이 있으면 로컬에서 먼저 읽어옴 (네트워크 오류 방지)
    if os.path.exists(LOCAL_FILE):
        return pd.read_csv(LOCAL_FILE, encoding="utf-8")
    else:
        # 로컬에 파일이 없으면 온라인에서 가져옴
        return pd.read_csv(REMOTE_URL, encoding="utf-8")

try:
    df_raw = load_data()
    total_count = len(df_raw)
    
    # 3. 데이터 전처리 및 예외 처리
    # 필수 4가지 속성에 결측치가 있거나 첫 주 관객(first_week_audi)이 0인 영화 제외
    required_cols = ['first_scrn', 'total_audi', 'days_in_top10', 'first_week_audi']
    df = df_raw.dropna(subset=required_cols).copy()
    df = df[df['first_week_audi'] > 0].copy()
    
    # 속성 변환 및 파생변수 생성
    df['스크린 수(로그)'] = np.log10(df['first_scrn'])
    df['누적 관객(로그)'] = np.log10(df['total_audi'])
    df['10위권 일수'] = df['days_in_top10']
    
    # 롱런 지수 생성 후 20 초과 값은 20으로 자르기 (clip)
    df['롱런 지수'] = df['total_audi'] / df['first_week_audi']
    df['롱런 지수'] = df['롱런 지수'].clip(upper=20)
    
    valid_count = len(df)
    
    # 전체 편수와 분석에 사용된 묶은 편수를 화면에 한 줄로 출력
    st.write(f"📊 **전체 영화 편수:** {total_count}편 | **분석 및 묶음에 사용된 영화 편수:** {valid_count}편")
    
    # 4. 분석 속성 선택 컴포넌트 (화면 상단 배치, 기본은 4개 모두 선택)
    feature_options = ["스크린 수(로그)", "누적 관객(로그)", "10위권 일수", "롱런 지수"]
    
    st.subheader("⚙️ 분석 설정")
    selected_features = st.multiselect(
        "군집 분석(묶기)에 사용할 속성을 선택하세요 (둘 이상 선택 필수)",
        options=feature_options,
        default=feature_options
    )
    
    # 2개 이상 선택 검증
    if len(selected_features) < 2:
        st.error("⚠️ 속성을 최소 2개 이상 선택해야 분석을 진행할 수 있습니다.")
    else:
        # 5. 데이터 표준화 및 K-평균 군집화 (세 묶음)
        X = df[selected_features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 재실행 시 결과 고정을 위한 난수 설정 (random_state=42)
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['cluster_raw'] = kmeans.fit_predict(X_scaled)
        
        # 누적 관객 평균이 큰 묶음부터 ㉮, ㉯, ㉰ 부여
        cluster_means = df.groupby('cluster_raw')['total_audi'].mean().sort_values(ascending=False)
        rank_mapping = {raw_idx: label for raw_idx, label in zip(cluster_means.index, ['㉮', '㉯', '㉰'])}
        df['묶음'] = df['cluster_raw'].map(rank_mapping)
        
        # ㉮, ㉯, ㉰ 순서로 데이터 정렬
        df = df.sort_values(by='묶음')
        
        # 6. Plotly 2차원 산점도 시각화
        st.subheader("📈 2차원 산점도 시각화")
        col1, col2 = st.columns(2)
        with col1:
            x_axis_2d = st.selectbox("2차원 가로축(X) 속성 선택", options=selected_features, index=0)
        with col2:
            y_axis_2d = st.selectbox("2차원 세로축(Y) 속성 선택", options=selected_features, index=min(1, len(selected_features)-1))
            
        fig_2d = px.scatter(
            df, 
            x=x_axis_2d, 
            y=y_axis_2d, 
            color='묶음',
            category_orders={"묶음": ["㉮", "㉯", "㉰"]},
            hover_name='movieNm',
            title=f"2차원 산점도: {x_axis_2d} vs {y_axis_2d}",
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        st.plotly_chart(fig_2d, use_container_width=True)
        
        # 7. Plotly 3차원 산점도 시각화 (속성이 셋보다 적으면 안내 문구 표시)
        st.subheader("🥽 3차원 산점도 시각화")
        if len(selected_features) < 3:
            st.info("💡 선택된 속성이 3개 미만입니다. 3차원 산점도를 그리려면 위 설정에서 속성을 3개 이상 선택해 주세요.")
        else:
            col_x, col_y, col_z = st.columns(3)
            with col_x:
                x_axis_3d = st.selectbox("3차원 X축 속성 선택", options=selected_features, index=0)
            with col_y:
                y_axis_3d = st.selectbox("3차원 Y축 속성 선택", options=selected_features, index=1)
            with col_z:
                z_axis_3d = st.selectbox("3차원 Z축 속성 선택", options=selected_features, index=2)
                
            fig_3d = px.scatter_3d(
                df,
                x=x_axis_3d,
                y=y_axis_3d,
                z=z_axis_3d,
                color='묶음',
                category_orders={"묶음": ["㉮", "㉯", "㉰"]},
                hover_name='movieNm',
                title=f"3차원 산점도: {x_axis_3d} x {y_axis_3d} x {z_axis_3d}",
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            fig_3d.update_traces(marker=dict(size=4))  # 마우스 회전이 원활하도록 점 크기를 작게 조절
            st.plotly_chart(fig_3d, use_container_width=True)
            
        # 8. 묶음별 통계 표 출력 (원래 단위 기준)
        st.subheader("📋 묶음별 요약 통계 및 대표 영화")
        
        stat_df = df.groupby('묶음').agg(
            편수=('movieCd', 'count'),
            스크린수_평균=('first_scrn', 'mean'),
            누적관객_평균=('total_audi', 'mean'),
            _10위권일수_평균=('days_in_top10', 'mean'),
            롱런지수_평균=('롱런 지수', 'mean')
        ).reset_index()
        
        stat_df.columns = ['묶음', '편수', '스크린 수 평균', '누적 관객 평균', '10위권 일수 평균', '롱런 지수 평균']
        st.dataframe(stat_df.style.format({
            '스크린 수 평균': '{:,.1f}',
            '누적 관객 평균': '{:,.1f}',
            '10위권 일수 평균': '{:,.1f}',
            '롱런 지수 평균': '{:,.2f}'
        }), use_container_width=True)
        
        # 9. 묶음별 누적 관객 많은 순 top 5 영화 제목 출력
        st.write("🔝 **묶음별 누적 관객 상위 5편 영화**")
        cols_clusters = st.columns(3)
        
        for idx, cluster_label in enumerate(['㉮', '㉯', '㉰']):
            with cols_clusters[idx]:
                st.markdown(f"### 묶음 {cluster_label}")
                cluster_movies = df[df['묶음'] == cluster_label].sort_values(by='total_audi', ascending=False).head(5)
                if not cluster_movies.empty:
                    for rank, row in enumerate(cluster_movies.itertuples(), 1):
                        st.write(f"{rank}. **{row.movieNm}** ({row.total_audi:,}명)")
                else:
                    st.write("해당 묶음에 영화가 존재하지 않습니다.")

except Exception as e:
    st.error(f"데이터 로드 또는 분석 과정 중 오류가 발생했습니다: {e}")
