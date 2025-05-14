from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from urllib.parse import unquote
import re
from underthesea import word_tokenize
from datetime import datetime
from sqlalchemy import func
from collections import Counter, defaultdict
from vncorenlp import VnCoreNLP
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

# Khởi tạo VnCoreNLP
rdrsegmenter = VnCoreNLP(
    "./VnCoreNLP-1.2/VnCoreNLP-1.1.1.jar", 
    annotators="wseg,pos,ner", 
    max_heap_size='-Xmx2g'
)

# Cấu hình kết nối MariaDB
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:220201@localhost:3306/vnexpress'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Model cho bảng all_articles_01
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

# Model cho bảng vietnamese_dictionary
class VietnameseDictionary(db.Model):
    __tablename__ = 'vietnamese_dictionary'
    id = db.Column(db.Integer, primary_key=True)
    article_id = db.Column(db.String(255), nullable=False)
    keyword = db.Column(db.String(255), nullable=False)
    part_of_speech = db.Column(db.String(50))
    english_equivalent = db.Column(db.String(255))
    definition = db.Column(db.Text)
    frequency = db.Column(db.Integer, default=1)

# Model cho bảng entity
class Entity(db.Model):
    __tablename__ = 'entity'
    id = db.Column(db.Integer, primary_key=True)
    article_id = db.Column(db.String(255), nullable=False)
    entity = db.Column(db.String(255), nullable=False)
    entity_type = db.Column(db.String(50), nullable=False)

# Model cho bảng entity_aliases
class EntityAlias(db.Model):
    __tablename__ = 'entity_aliases'
    id = db.Column(db.Integer, primary_key=True)
    article_id = db.Column(db.String(255), nullable=False)
    entity = db.Column(db.String(255), nullable=False)
    entity_alias = db.Column(db.String(255), nullable=False)
# Hàm lấy định nghĩa tiếng Việt từ Wiktionary
def get_vi_meaning(word):
    url = f"https://vi.wiktionary.org/wiki/{word.replace(' ', '_')}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        definition_list = soup.select("ol li")
        if definition_list:
            full_text = definition_list[0].text
            first_sentence = full_text.split('.')[0].strip() + '.'
            return first_sentence
        return "Không tìm thấy định nghĩa."
    except requests.RequestException:
        return "Lỗi khi truy vấn định nghĩa."

# Hàm xác định loại từ từ Wiktionary
def get_word_type(word):
    url = f"https://vi.wiktionary.org/wiki/{word.replace(' ', '_')}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return word, "Không thể truy cập trang."

        soup = BeautifulSoup(response.text, "html.parser")
        word_type_section = (
            soup.find("h3", id="Danh_từ") or
            soup.find("h3", id="Động_từ") or
            soup.find("h3", id="Tính_từ") or
            soup.find("h3", id="Đại_từ") or
            soup.find("h3", id="Trạng_từ") or
            soup.find("h3", id="Giới_từ") or
            soup.find("h3", id="Số_từ") or
            soup.find("h3", id="Liên_từ")
        )
        
        if word_type_section:
            word_type = word_type_section.get("id").replace("_", " ")
            return word, word_type
        else:
            word_type_span = soup.select_one("i, span")
            if word_type_span and word_type_span.text in ["danh từ", "động từ", "tính từ", "đại từ", "trạng từ", "giới từ", "số từ", "liên từ"]:
                return word, word_type_span.text.capitalize()
        
        return word, "Không tìm thấy loại từ."
    except requests.RequestException:
        return word, "Lỗi khi truy vấn loại từ."

# Hàm dịch tiếng Việt sang tiếng Anh
def translate_vi_to_en(text):
    url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=vi&tl=en&dt=t"
    params = {"q": text}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        result = response.json()
        translated_text = result[0][0][0]
        return translated_text
    except requests.RequestException as e:
        return f"Lỗi khi dịch: {e}"

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
    article_id = Article.query.filter_by(title=title).first().id if Article.query.filter_by(title=title).first() else 'unknown'
    return render_template('profile.html', title=title, author=author, pub_date=pub_date, content=content, article_url=article_url, article_id=article_id)

@app.route('/analyst')
def analyst():
    title = unquote(request.args.get('title', 'No Title'))
    content = unquote(request.args.get('content', 'No Content Available'))
    article_id = unquote(request.args.get('article_id', 'unknown'))
    return render_template('analyst.html', title=title, content=content, article_id=article_id)

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
    article_id = data.get("article_id", "unknown")
    full_text = f"{title} {content}".lower()  # Chuẩn hóa thành chữ thường

    # Lấy toàn bộ danh sách keyword từ vietnamese_dictionary (không giới hạn article_id)
    dictionary_words = set(word.keyword.lower() for word in VietnameseDictionary.query.all())

    # Tách từ đơn từ nội dung (giữ nguyên logic hiện tại)
    words = re.findall(r'[a-zA-ZÀ-ỹ]+', full_text)
    
    # Lọc các từ không có trong vietnamese_dictionary
    new_words = [word for word in words if word not in dictionary_words]
    
    # Tính tần suất các từ mới
    counter = Counter(new_words)
    word_frequencies = [[word, count] for word, count in counter.items()]
    
    result = {
        "single_word_count": len(counter),
        "single_word_frequencies": word_frequencies,
        "article_id": article_id  # Chỉ lưu article_id, không dùng để so sánh
    }
    return jsonify(result)

@app.route('/api/analyze/compound-words', methods=['POST'])
def analyze_compound_words():
    data = request.json
    title = data.get("title", "")
    content = data.get("content", "")
    article_id = data.get("article_id", "unknown")
    full_text = f"{title} {content}"

    # Lấy danh sách từ ghép từ vietnamese_dictionary
    dictionary_words = [word.keyword.lower() for word in VietnameseDictionary.query.all()]

    # Tách từ với ưu tiên từ trong vietnamese_dictionary
    def custom_tokenize(text):
        tokens = []
        text_lower = text.lower()
        i = 0
        while i < len(text_lower):
            matched = False
            # Kiểm tra các từ trong từ điển có khớp tại vị trí hiện tại
            for dict_word in dictionary_words:
                if text_lower.startswith(dict_word, i):
                    tokens.append(dict_word)
                    i += len(dict_word)
                    matched = True
                    break
            if not matched:
                # Nếu không khớp, lấy ký tự tiếp theo
                tokens.append(text_lower[i])
                i += 1
        return tokens

    # Tách từ ban đầu để xác định ranh giới từ
    initial_tokens = word_tokenize(full_text.lower())
    # Gộp lại các từ dựa trên từ điển
    final_tokens = []
    for token in initial_tokens:
        if ' ' in token:
            # Nếu là từ ghép, kiểm tra trực tiếp
            final_tokens.append(token)
        else:
            # Nếu là từ đơn, thử gộp với từ điển
            sub_tokens = custom_tokenize(token)
            final_tokens.extend(sub_tokens)

    # Lọc chỉ lấy từ ghép (có dấu cách) và không có trong vietnamese_dictionary
    compound_words = [token for token in final_tokens if ' ' in token and token.lower() not in dictionary_words]
    counter = Counter(compound_words)
    word_frequencies = [[word, count] for word, count in counter.items()]

    result = {
        "compound_word_count": len(counter),
        "compound_word_frequencies": word_frequencies,
        "article_id": article_id
    }
    return jsonify(result)

@app.route('/api/save_compound_words', methods=['POST'])
def save_compound_words():
    data = request.json
    title = data.get("title", "")
    words = data.get("words", [])

    try:
        article = Article.query.filter_by(title=title).first()
        article_id = str(article.id) if article else "unknown"

        for word_data in words:
            keyword = word_data.get("keyword")
            frequency = word_data.get("frequency", 1)

            # Lấy part_of_speech từ Wiktionary
            _, part_of_speech = get_word_type(keyword)

            # Lấy english_equivalent
            english_equivalent = translate_vi_to_en(keyword)

            # Lấy definition
            definition = get_vi_meaning(keyword)

            existing_word = VietnameseDictionary.query.filter_by(keyword=keyword, article_id=article_id).first()
            if existing_word:
                existing_word.frequency = frequency
                existing_word.part_of_speech = part_of_speech
                existing_word.english_equivalent = english_equivalent
                existing_word.definition = definition
            else:
                new_word = VietnameseDictionary(
                    article_id=article_id,
                    keyword=keyword,
                    part_of_speech=part_of_speech,
                    english_equivalent=english_equivalent,
                    definition=definition,
                    frequency=frequency
                )
                db.session.add(new_word)

        db.session.commit()
        return jsonify({"success": True})
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)})

# @app.route('/api/analyze/entities', methods=['POST'])
# def analyze_entities():
#     data = request.json
#     title = data.get("title", "")
#     content = data.get("content", "")
#     full_text = f"{title} {content}"

#     output = rdrsegmenter.annotate(full_text)
#     valid_labels = {"PER", "LOC", "ORG", "MISC"}
#     entities = []

#     for sentence in output["sentences"]:
#         current_entity = ""
#         current_label = ""
#         for word in sentence:
#             label = word["nerLabel"]
#             if label.startswith("B-"):
#                 if current_entity:
#                     entities.append((current_entity.replace("_", " "), current_label))
#                 current_entity = word["form"]
#                 current_label = label[2:]
#             elif label.startswith("I-") and label[2:] == current_label:
#                 current_entity += " " + word["form"]
#             else:
#                 if current_entity:
#                     entities.append((current_entity.replace("_", " "), current_label))
#                     current_entity = ""
#                     current_label = ""
#         if current_entity:
#             entities.append((current_entity.replace("_", " "), current_label))

#     filtered_entities = [(ent, label) for ent, label in entities if label in valid_labels]
#     grouped = defaultdict(list)
#     for ent, label in filtered_entities:
#         if ent not in grouped[label]:
#             grouped[label].append(ent)

#     return jsonify(grouped)

@app.route('/api/get_dictionary_words', methods=['GET'])
def get_dictionary_words():
    article_id = request.args.get('article_id', '')
    try:
        query = VietnameseDictionary.query
        if article_id and article_id != 'unknown':
            query = query.filter_by(article_id=article_id)
        dictionary_words = [word.keyword.lower() for word in query.all()]  # Chuẩn hóa chữ thường
        return jsonify({"success": True, "words": dictionary_words})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/update_content', methods=['POST'])
def update_content():
    data = request.json
    title = data.get("title", "")
    content = data.get("content", "")

    try:
        article = Article.query.filter_by(title=title).first()
        if not article:
            return jsonify({"success": False, "error": "Bài viết không tồn tại."})
        
        article.content = content
        db.session.commit()
        return jsonify({"success": True})
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)})

from urllib.parse import quote

# Route cho trang thêm từ ghép thủ công
@app.route('/add-compound-word')
def add_compound_word():
    title = unquote(request.args.get('title', 'No Title'))
    content = unquote(request.args.get('content', 'No Content Available'))
    
    # Lấy article_id từ cơ sở dữ liệu dựa trên title
    article = Article.query.filter_by(title=title).first()
    article_id = str(article.id) if article else 'unknown'

    return render_template(
        'add_compound_word.html',
        title=title,
        content=content,
        article_id=article_id
    )


@app.route('/api/manual_add_compound_word', methods=['POST'])
def manual_add_compound_word():
    data = request.json
    keyword = data.get("keyword", "").strip().lower()  # Chuyển thành chữ thường
    article_id = data.get("article_id", "").strip()
    title = data.get("title", "").strip()
    content = data.get("content", "").strip()

    if not keyword or not article_id or article_id == "unknown":
        return jsonify({"success": False, "error": "Thiếu hoặc không hợp lệ: từ khóa hoặc article_id"})

    try:
        article = Article.query.filter_by(id=article_id).first()
        if not article:
            return jsonify({"success": False, "error": "Không tìm thấy bài viết với article_id này."})

        # Tính tần suất xuất hiện của từ ghép
        full_text = f"{title} {content}".lower()
        frequency = len(re.findall(r'\b' + re.escape(keyword) + r'\b', full_text))

        # Kiểm tra xem từ đã tồn tại chưa
        existing = VietnameseDictionary.query.filter_by(keyword=keyword, article_id=article_id).first()
        if existing:
            return jsonify({"success": False, "error": "Từ này đã tồn tại."})

        # Lấy part_of_speech từ Wiktionary
        _, part_of_speech = get_word_type(keyword)

        # Lấy english_equivalent
        english_equivalent = translate_vi_to_en(keyword)

        # Lấy definition
        definition = get_vi_meaning(keyword)

        # Lưu từ vào cơ sở dữ liệu
        new_word = VietnameseDictionary(
            article_id=article_id,
            keyword=keyword,
            part_of_speech=part_of_speech,
            english_equivalent=english_equivalent,
            definition=definition,
            frequency=frequency if frequency > 0 else 1  # Đảm bảo tần suất ít nhất là 1
        )
        db.session.add(new_word)
        db.session.commit()
        return jsonify({"success": True, "frequency": frequency})
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)})

@app.route('/add-single-word')
def add_single_word():
    title = unquote(request.args.get('title', 'No Title'))
    content = unquote(request.args.get('content', 'No Content Available'))
    
    # Lấy article_id từ cơ sở dữ liệu dựa trên title
    article = Article.query.filter_by(title=title).first()
    article_id = str(article.id) if article else 'unknown'

    return render_template(
        'add_single_word.html',
        title=title,
        content=content,
        article_id=article_id
    )

@app.route('/api/save_single_words', methods=['POST'])
def save_single_words():
    data = request.json
    title = data.get("title", "")
    words = data.get("words", [])
    article_id = data.get("article_id", "unknown")

    try:
        article = Article.query.filter_by(title=title).first()
        article_id = str(article.id) if article else article_id

        for word_data in words:
            keyword = word_data.get("keyword").lower()  # Chuẩn hóa chữ thường
            frequency = word_data.get("frequency", 1)

            # Lấy part_of_speech từ Wiktionary
            _, part_of_speech = get_word_type(keyword)

            # Lấy english_equivalent
            english_equivalent = translate_vi_to_en(keyword)

            # Lấy definition
            definition = get_vi_meaning(keyword)

            existing_word = VietnameseDictionary.query.filter_by(keyword=keyword, article_id=article_id).first()
            if existing_word:
                existing_word.frequency = frequency
                existing_word.part_of_speech = part_of_speech
                existing_word.english_equivalent = english_equivalent
                existing_word.definition = definition
            else:
                new_word = VietnameseDictionary(
                    article_id=article_id,
                    keyword=keyword,
                    part_of_speech=part_of_speech,
                    english_equivalent=english_equivalent,
                    definition=definition,
                    frequency=frequency
                )
                db.session.add(new_word)

        db.session.commit()
        return jsonify({"success": True})
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/manual_add_single_word', methods=['POST'])
def manual_add_single_word():
    data = request.json
    keyword = data.get("keyword", "").strip().lower()  # Chuyển thành chữ thường
    article_id = data.get("article_id", "").strip()
    title = data.get("title", "").strip()
    content = data.get("content", "").strip()

    if not keyword or not article_id or article_id == "unknown":
        return jsonify({"success": False, "error": "Thiếu hoặc không hợp lệ: từ khóa hoặc article_id"})

    try:
        article = Article.query.filter_by(id=article_id).first()
        if not article:
            return jsonify({"success": False, "error": "Không tìm thấy bài viết với article_id này."})

        # Tính tần suất xuất hiện của từ đơn
        full_text = f"{title} {content}".lower()
        frequency = len(re.findall(r'\b' + re.escape(keyword) + r'\b', full_text))

        # Kiểm tra xem từ đã tồn tại chưa
        existing = VietnameseDictionary.query.filter_by(keyword=keyword, article_id=article_id).first()
        if existing:
            return jsonify({"success": False, "error": "Từ này đã tồn tại."})

        # Lấy part_of_speech từ Wiktionary
        _, part_of_speech = get_word_type(keyword)

        # Lấy english_equivalent
        english_equivalent = translate_vi_to_en(keyword)

        # Lấy definition
        definition = get_vi_meaning(keyword)

        # Lưu từ vào cơ sở dữ liệu
        new_word = VietnameseDictionary(
            article_id=article_id,
            keyword=keyword,
            part_of_speech=part_of_speech,
            english_equivalent=english_equivalent,
            definition=definition,
            frequency=frequency if frequency > 0 else 1  # Đảm bảo tần suất ít nhất là 1
        )
        db.session.add(new_word)
        db.session.commit()
        return jsonify({"success": True, "frequency": frequency})
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/get_tokens_for_highlight', methods=['POST'])
def get_tokens_for_highlight():
    data = request.json
    content = data.get("content", "")
    article_id = data.get("article_id", "")
    content_lower = content.lower()

    try:
        # Lấy danh sách từ khóa từ vietnamese_dictionary
        dict_words = VietnameseDictionary.query.with_entities(VietnameseDictionary.keyword).all()
        dict_keywords = [kw[0].lower() for kw in dict_words]

        # Sắp xếp từ khóa theo độ dài giảm dần
        dict_keywords.sort(key=lambda x: len(x), reverse=True)

        tokens = []
        processed_positions = set()  # Theo dõi các vị trí đã được xử lý

        for keyword in dict_keywords:
            # Chỉ khớp từ đúng (ví dụ: "kho" không khớp "khoảng")
            pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
            for match in pattern.finditer(content_lower):
                start, end = match.start(), match.end()
                # Kiểm tra nếu vị trí này chưa được xử lý
                if not any(start <= pos < end for pos in processed_positions):
                    tokens.append({
                        "token": match.group(),
                        "start": start,
                        "end": end
                    })
                    # Thêm các vị trí đã xử lý
                    processed_positions.update(range(start, end))

        return jsonify({"success": True, "tokens": tokens})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
    
@app.route('/api/analyze/entities', methods=['POST'])
def analyze_entities():
    data = request.json
    title = data.get("title", "")
    content = data.get("content", "")
    article_id = data.get("article_id", "unknown")
    full_text = f"{title} {content}"

    # Hàm chuẩn hóa văn bản
    def normalize_text(text):
        return re.sub(r'\s+', ' ', text.strip()).lower()

    # Lấy toàn bộ danh sách từ khóa từ vietnamese_dictionary (không lọc theo article_id)
    existing_keywords = set(
        normalize_text(word.keyword) for word in VietnameseDictionary.query.all()
    )

    # Lấy danh sách các thực thể đã lưu trong bảng entity cho bài viết này
    existing_entities = Entity.query.filter_by(article_id=article_id).all()
    entity_dict = {normalize_text(ent.entity): ent.entity_type for ent in existing_entities}

    # Tìm và thay thế các thực thể đã lưu trong nội dung
    processed_text = full_text.lower()
    entity_positions = []
    for entity in entity_dict.keys():
        pattern = r'\b' + re.escape(entity) + r'\b'
        matches = [(match.start(), match.end(), entity, entity_dict[entity]) for match in re.finditer(pattern, processed_text)]
        entity_positions.extend(matches)

    # Sắp xếp theo vị trí để xử lý từ cuối về đầu
    entity_positions.sort(key=lambda x: x[0], reverse=True)

    # Tạo danh sách các đoạn văn bản không chứa thực thể đã lưu
    segments = []
    last_end = len(processed_text)
    preserved_entities = []
    for start, end, entity, entity_type in entity_positions:
        if start > last_end:
            continue
        # Lưu đoạn văn bản từ cuối đến vị trí hiện tại
        segments.insert(0, full_text[last_end:])
        # Lưu thực thể đã tìm thấy
        preserved_entities.insert(0, (entity, entity_type))
        segments.insert(0, full_text[start:end])
        last_end = start
    if last_end > 0:
        segments.insert(0, full_text[:last_end])

    # Phân tích các đoạn văn bản còn lại bằng VnCoreNLP
    entities = []
    segment_offset = 0
    for i, segment in enumerate(segments):
        if i % 2 == 1:  # Đây là thực thể đã lưu
            entity, entity_type = preserved_entities[i // 2]
            entities.append((entity, entity_type))
            segment_offset += len(segment)
            continue

        if not segment.strip():
            continue

        output = rdrsegmenter.annotate(segment)
        valid_labels = {"PER", "LOC", "ORG", "MISC"}

        current_entity = ""
        current_label = ""
        for sentence in output["sentences"]:
            for word in sentence:
                label = word["nerLabel"]
                if label.startswith("B-"):
                    if current_entity:
                        entities.append((normalize_text(current_entity.replace("_", " ")), current_label))
                    current_entity = word["form"]
                    current_label = label[2:]
                elif label.startswith("I-") and label[2:] == current_label:
                    current_entity += " " + word["form"]
                else:
                    if current_entity:
                        entities.append((normalize_text(current_entity.replace("_", " ")), current_label))
                        current_entity = ""
                        current_label = ""
            if current_entity:
                entities.append((normalize_text(current_entity.replace("_", " ")), current_label))
                current_entity = ""
                current_label = ""

    # Lọc các thực thể hợp lệ và loại bỏ các từ đã tồn tại trong vietnamese_dictionary
    filtered_entities = [
        (ent, label) for ent, label in entities 
        if label in valid_labels and ent not in existing_keywords
    ]

    # Nhóm và tính tần suất
    grouped = defaultdict(list)
    frequency_counter = defaultdict(Counter)

    for ent, label in filtered_entities:
        frequency_counter[label][ent] += 1
        if ent not in grouped[label]:
            grouped[label].append({"entity": ent, "frequency": frequency_counter[label][ent]})

    result = {
        "article_id": article_id,
        "entities": {
            label: grouped[label]
            for label in valid_labels if grouped[label]
        }
    }

    return jsonify(result)

@app.route('/api/save_entities', methods=['POST'])
def save_entities():
    data = request.json
    title = data.get("title", "")
    entities = data.get("entities", [])
    article_id = data.get("article_id", "unknown")

    try:
        article = Article.query.filter_by(title=title).first()
        article_id = str(article.id) if article else article_id

        for entity_data in entities:
            entity_name = entity_data.get("entity").strip().lower()  # Chuẩn hóa thành chữ thường
            frequency = entity_data.get("frequency", 1)
            entity_type = entity_data.get("entity_type")

            # Lưu vào vietnamese_dictionary
            existing_word = VietnameseDictionary.query.filter_by(keyword=entity_name, article_id=article_id).first()
            if not existing_word:
                new_word = VietnameseDictionary(
                    article_id=article_id,
                    keyword=entity_name,
                    part_of_speech=entity_type,
                    frequency=frequency
                )
                db.session.add(new_word)

            # Lưu vào entity
            existing_entity = Entity.query.filter_by(entity=entity_name, article_id=article_id, entity_type=entity_type).first()
            if not existing_entity:
                new_entity = Entity(
                    article_id=article_id,
                    entity=entity_name,
                    entity_type=entity_type
                )
                db.session.add(new_entity)

        db.session.commit()
        return jsonify({"success": True})
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)})

@app.route('/add-entity')
def add_entity():
    title = unquote(request.args.get('title', 'No Title'))
    content = unquote(request.args.get('content', 'No Content Available'))
    
    # Lấy article_id từ cơ sở dữ liệu dựa trên title
    article = Article.query.filter_by(title=title).first()
    article_id = str(article.id) if article else 'unknown'

    return render_template(
        'add_entity.html',
        title=title,
        content=content,
        article_id=article_id
    )

@app.route('/api/manual_add_entity', methods=['POST'])
def manual_add_entity():
    data = request.json
    entity = data.get("entity", "").strip().lower()  # Chuyển thành chữ thường
    entity_alias = data.get("entity_alias", "").strip().lower()  # Chuyển thành chữ thường
    article_id = data.get("article_id", "").strip()
    title = data.get("title", "").strip()
    content = data.get("content", "").strip()

    if not entity or not article_id or article_id == "unknown":
        return jsonify({"success": False, "error": "Thiếu hoặc không hợp lệ: thực thể hoặc article_id"})

    try:
        article = Article.query.filter_by(id=article_id).first()
        if not article:
            return jsonify({"success": False, "error": "Không tìm thấy bài viết với article_id này."})

        # Tính tần suất xuất hiện của entity và entity_alias
        full_text = f"{title} {content}".lower()
        entity_frequency = len(re.findall(r'\b' + re.escape(entity) + r'\b', full_text))
        entity_alias_frequency = len(re.findall(r'\b' + re.escape(entity_alias) + r'\b', full_text)) if entity_alias else 0

        # Kiểm tra xem entity đã tồn tại trong vietnamese_dictionary chưa
        existing_word = VietnameseDictionary.query.filter_by(keyword=entity, article_id=article_id).first()
        if existing_word:
            return jsonify({"success": False, "error": "Thực thể này đã tồn tại trong từ điển."})

        # Lấy part_of_speech, english_equivalent, definition cho entity
        _, entity_pos = get_word_type(entity)
        entity_english = translate_vi_to_en(entity)
        entity_definition = get_vi_meaning(entity)

        # Lưu entity vào vietnamese_dictionary
        new_word = VietnameseDictionary(
            article_id=article_id,
            keyword=entity,
            part_of_speech=entity_pos,
            english_equivalent=entity_english,
            definition=entity_definition,
            frequency=entity_frequency if entity_frequency > 0 else 1
        )
        db.session.add(new_word)

        # Nếu có entity_alias, lưu vào vietnamese_dictionary
        if entity_alias:
            # Kiểm tra xem entity_alias đã tồn tại trong vietnamese_dictionary chưa
            existing_alias_word = VietnameseDictionary.query.filter_by(keyword=entity_alias, article_id=article_id).first()
            if existing_alias_word:
                return jsonify({"success": False, "error": "Bí danh thực thể này đã tồn tại trong từ điển."})

            # Lấy part_of_speech, english_equivalent, definition cho entity_alias
            _, alias_pos = get_word_type(entity_alias)
            alias_english = translate_vi_to_en(entity_alias)
            alias_definition = get_vi_meaning(entity_alias)

            # Lưu entity_alias vào vietnamese_dictionary
            new_alias_word = VietnameseDictionary(
                article_id=article_id,
                keyword=entity_alias,
                part_of_speech=alias_pos,
                english_equivalent=alias_english,
                definition=alias_definition,
                frequency=entity_alias_frequency if entity_alias_frequency > 0 else 1
            )
            db.session.add(new_alias_word)

        # Lưu vào entity (giả sử entity_type là MISC, có thể yêu cầu người dùng chọn sau)
        existing_entity = Entity.query.filter_by(entity=entity, article_id=article_id).first()
        if not existing_entity:
            new_entity = Entity(
                article_id=article_id,
                entity=entity,
                entity_type="MISC"  # Mặc định là MISC, có thể mở rộng
            )
            db.session.add(new_entity)

        # Lưu vào entity_aliases nếu có entity_alias
        if entity_alias:
            existing_alias = EntityAlias.query.filter_by(entity=entity, entity_alias=entity_alias, article_id=article_id).first()
            if not existing_alias:
                new_alias = EntityAlias(
                    article_id=article_id,
                    entity=entity,
                    entity_alias=entity_alias
                )
                db.session.add(new_alias)

        db.session.commit()
        return jsonify({"success": True, "frequency": entity_frequency})
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)})

    
if __name__ == '__main__':
    app.run(debug=True)