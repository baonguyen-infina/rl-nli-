"""Bảng mã chủ đề 01–13 của uid ViANLI -> tên chủ đề (đã khớp số đếm Table 3).
"""
# Tên chủ đề dùng mã T01..T13 (quyết định của spec); đổi giá trị sang tên thật khi có Table 3.
TOPICS = {f"{i:02d}": f"T{i:02d}" for i in range(1, 14)}
