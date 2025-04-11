PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Streamlit 앱 실행
cd "${PROJECT_ROOT}/frontend"
streamlit run streamlit_app.py --server.port 8000