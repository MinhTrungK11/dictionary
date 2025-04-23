from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from urllib.parse import unquote
import re
from underthesea import word_tokenize
import uuid
from datetime import datetime
from sqlalchemy import func
from collections import Counter
from vncorenlp import VnCoreNLP



app = Flask(__name__)

rdrsegmenter = VnCoreNLP(
        "./VnCoreNLP-1.2/VnCoreNLP-1.1.1.jar", 
        annotators="wseg,pos,ner", 
        max_heap_size='-Xmx2g'
    )

# Cấu hình kết nối MariaDB
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://phu:123456@mystikos.net:3306/vnexpress'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class Article(db.Model):
    __tablename__ = 'all_articles_01'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    author = db.Column(db.String(100), nullable=False)
    pub_date = db.Column(db.Date, nullable=False)
    category_level1 = db.Column(db.String(100))
    category_level2 = db.Column(db.String(100))
    article_url = db.Column(db.String(255))  
    content = db.Column(db.Text) 
    publish_date = db.Column(db.Date, nullable=False)

    def to_dict(self):
        pub_date_str = self.pub_date.strftime('%Y-%m-%d') if isinstance(self.pub_date, datetime) else self.pub_date
        return {
            'id': self.id,
            'title': self.title,
            'author': self.author,
            'pub_date': pub_date_str,
            'category_level1': self.category_level1,
            'category_level2': self.category_level2,
            'article_url': self.article_url,
            'content': self.content,
            'publish_date': self.publish_date
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/articles')
def articles():
    page = request.args.get('page', 1, type=int)
    per_page = 10
    search = request.args.get('search', '').strip()
    category_level1 = request.args.get('category_level1', '').strip()
    pub_date_from = request.args.get('pub_date_from', '').strip()
    pub_date_to = request.args.get('pub_date_to', '').strip()

    query = Article.query

    if search:
        keyword = f"%{search}%"
        query = query.filter(
            db.or_(
                Article.title.ilike(keyword),
                Article.author.ilike(keyword)
            )
        )

    if category_level1:
        query = query.filter(Article.category_level1 == category_level1)

    from datetime import datetime

    def parse_date(vn_date_str):
        try:
            return datetime.strptime(vn_date_str, "Thứ %A, %d/%m/%Y, %H:%M (GMT+7)")
        except:
            return None

    if pub_date_from:
        try:
            from_date = datetime.strptime(pub_date_from, "%Y-%m-%d").date()
            query = query.filter(Article.publish_date >= from_date)
        except:
            pass

    if pub_date_to:
        try:
            to_date = datetime.strptime(pub_date_to, "%Y-%m-%d").date()
            query = query.filter(Article.publish_date <= to_date)
        except:
            pass

    pagination = query.order_by(Article.id.asc()).paginate(page=page, per_page=per_page, error_out=False)
    articles = [a.to_dict() for a in pagination.items]

    # Lấy tất cả các giá trị duy nhất cho danh mục
    all_categories = [c[0] for c in db.session.query(Article.category_level1).distinct() if c[0]]

    return render_template(
        'articles.html',
        transactions=articles,
        pagination=pagination,
        search=search,
        category_level1=category_level1,
        pub_date_from=pub_date_from,
        pub_date_to=pub_date_to,
        all_categories=all_categories
    )

@app.route('/profile')
def profile():
    title = unquote(request.args.get('title', 'No Title'))
    author = unquote(request.args.get('author', 'Unknown Author'))
    pub_date = unquote(request.args.get('pub_date', 'Unknown Date'))
    content = unquote(request.args.get('content', 'No Content Available'))
    article_url = unquote(request.args.get('article_url', '#'))
    return render_template('profile.html', title=title, author=author, pub_date=pub_date, content=content, article_url=article_url)

@app.route('/analyst')
def analyst():
    title = unquote(request.args.get('title', 'No Title'))
    content = unquote(request.args.get('content', 'No Content Available'))
    return render_template('analyst.html', title=title, content=content)

@app.route('/api/analyze/basic', methods=['POST'])
def analyze_basic():
    data = request.json
    title = data.get("title", "")
    content = data.get("content", "")
    full_text = f"{title} {content}"
    words = re.findall(r'\w+', full_text.lower())
    result = {
        "total_characters": len(full_text),
        "total_words": len(words),
        "unique_words": len(set(words))
    }
    return jsonify(result)

@app.route('/api/analyze/special', methods=['POST'])
def analyze_special():
    data = request.json
    title = data.get("title", "")
    content = data.get("content", "")
    full_text = f"{title} {content}"
    specials = re.findall(r'[^\w\s]', full_text)
    frequencies = dict(Counter(specials))
    result = {
        "special_character_count": len(specials),
        "special_character_frequencies": frequencies
    }
    return jsonify(result)

@app.route('/api/analyze/single-words', methods=['POST'])
def analyze_single_words():
    data = request.json
    title = data.get("title", "")
    content = data.get("content", "")
    full_text = f"{title} {content}"
    # Tìm tất cả các từ, loại bỏ số và ký tự đặc biệt
    words = re.findall(r'[a-zA-ZÀ-ỹ]+', full_text.lower())
    # Giữ thứ tự xuất hiện và đếm tần suất
    seen = set()
    word_frequencies = []
    for word in words:
        if word not in seen:
            seen.add(word)
            count = words.count(word)
            word_frequencies.append([word, count])
    result = {
        "single_word_count": len(words),
        "single_word_frequencies": word_frequencies
    }
    return jsonify(result)

# Thêm API để phân tích từ ghép
@app.route('/api/analyze/compound-words', methods=['POST'])
def analyze_compound_words():
    from collections import Counter

    data = request.json
    title = data.get("title", "")
    content = data.get("content", "")
    full_text = f"{title} {content}"

    # Tách từ bằng underthesea
    tokens = word_tokenize(full_text.lower())

    # Lấy các token có chứa khoảng trắng → tức là từ ghép
    compound_words = [token for token in tokens if ' ' in token]

    # Đếm tần suất xuất hiện
    counter = Counter(compound_words)

    # Đưa ra kết quả dạng [từ ghép, số lần]
    word_frequencies = [[word, count] for word, count in counter.items()]

    result = {
        "compound_word_count": len(counter),  # số từ ghép duy nhất
        "compound_word_frequencies": word_frequencies
    }
    return jsonify(result)

@app.route('/api/analyze/entities', methods=['POST'])
def analyze_entities():
    data = request.json
    title = data.get("title", "")
    content = data.get("content", "")
    full_text = f"{title} {content}"

    output = rdrsegmenter.annotate(full_text)
    valid_labels = {"PER", "LOC", "ORG", "MISC"}
    entities = []

    for sentence in output["sentences"]:
        current_entity = ""
        current_label = ""
        for word in sentence:
            label = word["nerLabel"]
            if label.startswith("B-"):
                if current_entity:
                    entities.append((current_entity.replace("_", " "), current_label))
                current_entity = word["form"]
                current_label = label[2:]
            elif label.startswith("I-") and label[2:] == current_label:
                current_entity += " " + word["form"]
            else:
                if current_entity:
                    entities.append((current_entity.replace("_", " "), current_label))
                    current_entity = ""
                    current_label = ""
        if current_entity:
            entities.append((current_entity.replace("_", " "), current_label))

    filtered_entities = [(ent, label) for ent, label in entities if label in valid_labels]

    # Gom nhóm theo nhãn
    from collections import defaultdict
    grouped = defaultdict(list)
    for ent, label in filtered_entities:
        if ent not in grouped[label]:
            grouped[label].append(ent)

    return jsonify(grouped)



if __name__ == '__main__':
    app.run(debug=True)