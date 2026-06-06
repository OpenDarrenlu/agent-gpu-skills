#!/usr/bin/env python3
"""
知乎知识库搜索工具
根据用户查询从本地知乎文章库中检索相关文章
"""

import sqlite3
import os
import sys
import json
import re

DB_PATH = '/Users/moonshot/Documents/kimi/workspace/zhihu_knowledge_base/index/zhihu_articles.db'
ARTICLES_DIR = '/Users/moonshot/Documents/kimi/workspace/zhihu_knowledge_base/articles'


def search_articles(query, limit=10):
    """根据查询搜索相关文章"""
    if not os.path.exists(DB_PATH):
        print(f"错误: 数据库不存在: {DB_PATH}")
        return []
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    # 使用FTS5全文搜索
    try:
        c.execute('''
            SELECT a.id, a.title, a.author, a.url, a.content_length, a.content_preview, a.keywords, a.filepath
            FROM articles a
            JOIN articles_fts fts ON a.rowid = fts.rowid
            WHERE articles_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        ''', (query, limit))
        results = [dict(row) for row in c.fetchall()]
    except:
        # FTS搜索失败，回退到LIKE搜索
        c.execute('''
            SELECT id, title, author, url, content_length, content_preview, keywords, filepath
            FROM articles
            WHERE title LIKE ? OR content_preview LIKE ? OR keywords LIKE ?
            ORDER BY content_length DESC
            LIMIT ?
        ''', (f'%{query}%', f'%{query}%', f'%{query}%', limit))
        results = [dict(row) for row in c.fetchall()]
    
    conn.close()
    return results


def get_article_content(article_id):
    """获取文章完整内容"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT filepath FROM articles WHERE id = ?', (article_id,))
    row = c.fetchone()
    conn.close()
    
    if not row:
        return None
    
    filepath = row[0]
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def list_all_authors():
    """列出所有有文章的博主"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT author, author_id, COUNT(*) as article_count, SUM(content_length) as total_length
        FROM articles
        GROUP BY author
        ORDER BY article_count DESC
    ''')
    results = c.fetchall()
    conn.close()
    return results


def list_articles_by_author(author_name):
    """列出某位博主的所有文章"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('''
        SELECT id, title, url, content_length, content_preview, keywords
        FROM articles
        WHERE author = ?
        ORDER BY content_length DESC
    ''', (author_name,))
    results = [dict(row) for row in c.fetchall()]
    conn.close()
    return results


def get_stats():
    """获取知识库统计信息"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('SELECT COUNT(*) FROM articles')
    total_articles = c.fetchone()[0]
    
    c.execute('SELECT COUNT(DISTINCT author) FROM articles')
    total_authors = c.fetchone()[0]
    
    c.execute('SELECT SUM(content_length) FROM articles')
    total_chars = c.fetchone()[0] or 0
    
    conn.close()
    
    return {
        'total_articles': total_articles,
        'total_authors': total_authors,
        'total_chars': total_chars,
        'db_path': DB_PATH,
        'articles_dir': ARTICLES_DIR
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python3 zhihu_search.py <command> [args]")
        print("命令:")
        print("  search <query> [limit]  - 搜索文章")
        print("  content <article_id>    - 获取文章完整内容")
        print("  authors                 - 列出所有博主")
        print("  by-author <author>      - 列出博主的所有文章")
        print("  stats                   - 知识库统计")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == 'search':
        query = sys.argv[2] if len(sys.argv) > 2 else ''
        limit = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        results = search_articles(query, limit)
        print(json.dumps(results, ensure_ascii=False, indent=2))
    
    elif command == 'content':
        article_id = sys.argv[2] if len(sys.argv) > 2 else ''
        content = get_article_content(article_id)
        print(content or "文章未找到")
    
    elif command == 'authors':
        results = list_all_authors()
        for author, author_id, count, length in results:
            print(f"{author}: {count} 篇, {length} 字")
    
    elif command == 'by-author':
        author = sys.argv[2] if len(sys.argv) > 2 else ''
        results = list_articles_by_author(author)
        print(json.dumps(results, ensure_ascii=False, indent=2))
    
    elif command == 'stats':
        stats = get_stats()
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    
    else:
        print(f"未知命令: {command}")
