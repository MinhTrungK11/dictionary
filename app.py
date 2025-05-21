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
from transformer import Transformer, tokenizer_ipt, tokenizer_opt,create_masks
import tensorflow as tf
import tensorflow_datasets as tfds


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
    article_id = db.Column(db.String(255)) 
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
            'article_id': self.article_id,
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

# Model cho bảng undiacritic_sentences
class UndiacriticSentence(db.Model):
    __tablename__ = 'undiacritic_sentences'
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'content': self.content
        }

# Model cho bảng diacritic_sentences
class DiacriticSentence(db.Model):
    __tablename__ = 'diacritic_sentences'

    article_id = db.Column(db.Integer, primary_key=True)
    undiacritic = db.Column(db.Text, nullable=False) # Nội dung gốc không dấu
    content = db.Column(db.Text, nullable=False)       # Nội dung đã thêm dấu  

    def to_dict(self):
        return {
            'article_id': self.article_id,
            'undiacritic': self.undiacritic,
            'content': self.content
        }

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

@app.route('/transaction')
def transaction():
    page = request.args.get('page', 1, type=int)
    per_page = 10
    search = request.args.get('search', '').strip()

    query = DiacriticSentence.query

    if search:
        keyword = f"%{search}%"
        query = query.filter(
            db.or_(
                DiacriticSentence.undiacritic.ilike(keyword)
            )
        )

    pagination = query.order_by(DiacriticSentence.article_id.asc()).paginate(page=page, per_page=per_page, error_out=False)
    transaction = [a.to_dict() for a in pagination.items]


    return render_template(
        'sentences.html',
        transactions=transaction,
        pagination=pagination,
        search=search,
    )

@app.route('/profile')
def profile():
    title = unquote(request.args.get('title', 'No Title'))
    author = unquote(request.args.get('author', 'Unknown Author'))
    pub_date = unquote(request.args.get('pub_date', 'Unknown Date'))
    content = unquote(request.args.get('content', 'No Content Available'))
    article_url = unquote(request.args.get('article_url', '#'))
    article_id = Article.query.filter_by(title=title).first().article_id if Article.query.filter_by(title=title).first() else 'unknown'
    return render_template('profile.html', title=title, author=author, pub_date=pub_date, content=content, article_url=article_url, article_id=article_id)

@app.route('/transactiondetail')
def transactiondetail():
    undiacritic = unquote(request.args.get('undiacritic', 'No Content Available'))
    transaction = DiacriticSentence.query.filter_by(undiacritic=undiacritic).first()
    if transaction:
        transaction_id = transaction.article_id
        content = transaction.content
    else:
        transaction_id = 'unknown'
        content = 'No Content Available'
    return render_template('transaction_detail.html', undiacritic=undiacritic, content=content, article_id=transaction_id)


@app.route('/analyst')
def analyst():
    title = unquote(request.args.get('title', 'No Title'))
    content = unquote(request.args.get('content', 'No Content Available'))
    article_id = unquote(request.args.get('article_id', 'unknown'))
    return render_template('analyst.html', title=title, content=content, article_id=article_id)

@app.route('/analyst_tran')
def analyst_tran():
    content = unquote(request.args.get('content', 'No Content Available'))
    article_id = unquote(request.args.get('article_id', 'unknown'))
    return render_template('analyst_tran.html', content=content, article_id=article_id)

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

@app.route('/api/analyze/basic_tran', methods=['POST'])
def analyze_basic_tran():
    data = request.json
    content = data.get("content", "")
    words = re.findall(r'\w+', content.lower())
    result = {
        "total_characters": len(content),
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

@app.route('/api/analyze/special_tran', methods=['POST'])
def analyze_special_tran():
    data = request.json
    content = data.get("content", "")
    specials = re.findall(r'[^\w\s]', content)
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

@app.route('/api/analyze/single-words_tran', methods=['POST'])
def analyze_single_words_tran():
    data = request.json
    content = data.get("content", "")
    article_id = data.get("article_id", "unknown")
    content = content.lower() 

    dictionary_words = set(word.keyword.lower() for word in VietnameseDictionary.query.all())

    words = re.findall(r'[a-zA-ZÀ-ỹ]+', content)
    
    new_words = [word for word in words if word not in dictionary_words]
    
    counter = Counter(new_words)
    word_frequencies = [[word, count] for word, count in counter.items()]
    
    result = {
        "single_word_count": len(counter),
        "single_word_frequencies": word_frequencies,
        "article_id": article_id  
    }
    return jsonify(result)

@app.route('/api/analyze/compound-words', methods=['POST'])
def analyze_compound_words():
    data = request.json
    title = data.get("title", "")
    content = data.get("content", "")
    article_id = data.get("article_id", "unknown")
    full_text = f"{title} {content}"

    dictionary_words = sorted(
        [word.keyword.lower() for word in VietnameseDictionary.query.all()],
        key=lambda x: -len(x)  # Ưu tiên từ dài hơn
    )
    dictionary_set = set(dictionary_words)

    def custom_tokenize_by_words(words, dictionary_set, max_dict_len=5):
        tokens = []
        i = 0
        while i < len(words):
            matched = False
            for l in range(min(max_dict_len, len(words) - i), 0, -1):
                phrase = ' '.join(words[i:i+l])
                if phrase in dictionary_set:
                    tokens.append(phrase)
                    i += l
                    matched = True
                    break
            if not matched:
                tokens.append(words[i])
                i += 1
        return tokens

    words = word_tokenize(full_text.lower())
    tokens = custom_tokenize_by_words(words, dictionary_set)
    
    # Lọc ra từ ghép không nằm trong từ điển (ẩn những từ đã có trong dictionary)
    compound_words = [token for token in tokens if ' ' in token and token not in dictionary_set]

    counter = Counter(compound_words)
    word_frequencies = [[word, count] for word, count in counter.items()]

    result = {
        "compound_word_count": len(counter),
        "compound_word_frequencies": word_frequencies,
        "article_id": article_id
    }
    return jsonify(result)

@app.route('/api/analyze/compound-words_tran', methods=['POST'])
def analyze_compound_words_tran():
    data = request.json
    content = data.get("content", "")
    article_id = data.get("article_id", "unknown")

    dictionary_words = sorted(
        [word.keyword.lower() for word in VietnameseDictionary.query.all()],
        key=lambda x: -len(x) 
    )
    dictionary_set = set(dictionary_words)

    def custom_tokenize_by_words(words, dictionary_set, max_dict_len=5):
        tokens = []
        i = 0
        while i < len(words):
            matched = False
            for l in range(min(max_dict_len, len(words) - i), 0, -1):
                phrase = ' '.join(words[i:i+l])
                if phrase in dictionary_set:
                    tokens.append(phrase)
                    i += l
                    matched = True
                    break
            if not matched:
                tokens.append(words[i])
                i += 1
        return tokens

    words = word_tokenize(content.lower())
    tokens = custom_tokenize_by_words(words, dictionary_set)
    
    compound_words = [token for token in tokens if ' ' in token and token not in dictionary_set]

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
        article_id = str(article.article_id) if article else "unknown"

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

@app.route('/api/save_compound_words_tran', methods=['POST'])
def save_compound_words_tran():
    data = request.json
    content = data.get("content", "")
    words = data.get("words", [])

    try:
        article = DiacriticSentence.query.filter_by(content=content).first()
        article_id = str(article.article_id) if article else "unknown"

        for word_data in words:
            keyword = word_data.get("keyword")
            frequency = word_data.get("frequency", 1)

            _, part_of_speech = get_word_type(keyword)

            english_equivalent = translate_vi_to_en(keyword)

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

@app.route('/api/analyze/entities_tran', methods=['POST'])
def analyze_entities_tran():
    data = request.json
    content = data.get("content", "")
    article_id = data.get("article_id", "unknown")

    def normalize_text(text):
        return re.sub(r'\s+', ' ', text.strip()).lower()

    existing_keywords = set(
        normalize_text(word.keyword) for word in VietnameseDictionary.query.all()
    )

    existing_entities = Entity.query.filter_by(article_id=article_id).all()
    entity_dict = {normalize_text(ent.entity): ent.entity_type for ent in existing_entities}

    processed_text = content.lower()
    entity_positions = []
    for entity in entity_dict.keys():
        pattern = r'\b' + re.escape(entity) + r'\b'
        matches = [(match.start(), match.end(), entity, entity_dict[entity]) for match in re.finditer(pattern, processed_text)]
        entity_positions.extend(matches)

    entity_positions.sort(key=lambda x: x[0], reverse=True)

    segments = []
    last_end = len(processed_text)
    preserved_entities = []
    for start, end, entity, entity_type in entity_positions:
        if start > last_end:
            continue
        segments.insert(0, content[last_end:])
        preserved_entities.insert(0, (entity, entity_type))
        segments.insert(0, content[start:end])
        last_end = start
    if last_end > 0:
        segments.insert(0, content[:last_end])

    entities = []
    segment_offset = 0
    for i, segment in enumerate(segments):
        if i % 2 == 1:  
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

    filtered_entities = [
        (ent, label) for ent, label in entities 
        if label in valid_labels and ent not in existing_keywords
    ]

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
        article_id = str(article.article_id) if article else article_id

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
    
@app.route('/api/save_entities_tran', methods=['POST'])
def save_entities_tran():
    data = request.json
    content = data.get("content", "")
    entities = data.get("entities", [])
    article_id = data.get("article_id", "unknown")

    try:
        article = DiacriticSentence.query.filter_by(content=content).first()
        article_id = str(article.article_id) if article else article_id

        for entity_data in entities:
            entity_name = entity_data.get("entity").strip().lower() 
            frequency = entity_data.get("frequency", 1)
            entity_type = entity_data.get("entity_type")

            existing_word = VietnameseDictionary.query.filter_by(keyword=entity_name, article_id=article_id).first()
            if not existing_word:
                new_word = VietnameseDictionary(
                    article_id=article_id,
                    keyword=entity_name,
                    part_of_speech=entity_type,
                    frequency=frequency
                )
                db.session.add(new_word)

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

@app.route('/api/get_dictionary_words', methods=['GET'])
def get_dictionary_words():
    article_id = request.args.get('article_id', '')
    try:
        query = VietnameseDictionary.query
        if article_id and article_id != 'unknown':
            query = query.filter_by(article_id=article_id)
        dictionary_words = [word.keyword.lower() for word in query.all()]
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

@app.route('/api/update_content_tran', methods=['POST'])
def update_content_tran():
    data = request.json
    content = data.get("content", "")

    try:
        article = DiacriticSentence.query.filter_by(content=content).first()
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
    article_id = article.article_id if article else 'unknown'

    return render_template(
        'add_compound_word.html',
        title=title,
        content=content,
        article_id=article_id
    )

@app.route('/add-compound-word_tran')
def add_compound_word_tran():
    content = unquote(request.args.get('content', 'No Content Available'))
    
    article = DiacriticSentence.query.filter_by(content=content).first()
    article_id = article.article_id if article else 'unknown'

    return render_template(
        'add_compound_word_tran.html',
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
        article = Article.query.filter_by(article_id=article_id).first()
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

@app.route('/api/manual_add_compound_word_tran', methods=['POST'])
def manual_add_compound_word_tran():
    data = request.json
    keyword = data.get("keyword", "").strip().lower() 
    article_id = data.get("article_id", "").strip()
    content = data.get("content", "").strip()

    if not keyword or not article_id or article_id == "unknown":
        return jsonify({"success": False, "error": "Thiếu hoặc không hợp lệ: từ khóa hoặc article_id"})

    try:
        article = DiacriticSentence.query.filter_by(article_id=article_id).first()
        if not article:
            return jsonify({"success": False, "error": "Không tìm thấy bài viết với article_id này."})

        full_text = content.lower()
        frequency = len(re.findall(r'\b' + re.escape(keyword) + r'\b', full_text))

        existing = VietnameseDictionary.query.filter_by(keyword=keyword, article_id=article_id).first()
        if existing:
            return jsonify({"success": False, "error": "Từ này đã tồn tại."})

        _, part_of_speech = get_word_type(keyword)

        english_equivalent = translate_vi_to_en(keyword)

        definition = get_vi_meaning(keyword)
        
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
    article_id = article.article_id if article else 'unknown'

    return render_template(
        'add_single_word.html',
        title=title,
        content=content,
        article_id=article_id
    )

@app.route('/add-single-word_tran')
def add_single_word_tran():
    content = unquote(request.args.get('content', 'No Content Available'))
    
    article = DiacriticSentence.query.filter_by(content=content).first()
    article_id = article.article_id if article else 'unknown'

    return render_template(
        'add_single_word_tran.html',
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
        article_id = str(article.article_id) if article else article_id

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

@app.route('/api/save_single_words_tran', methods=['POST'])
def save_single_words_tran():
    data = request.json
    content = data.get("content", "")
    words = data.get("words", [])
    article_id = data.get("article_id", "unknown")

    try:
        article = DiacriticSentence.query.filter_by(content=content).first()
        article_id = str(article.article_id) if article else article_id

        for word_data in words:
            keyword = word_data.get("keyword").lower() 
            frequency = word_data.get("frequency", 1)

            _, part_of_speech = get_word_type(keyword)

            english_equivalent = translate_vi_to_en(keyword)

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
        article = Article.query.filter_by(article_id=article_id).first()
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
    
@app.route('/api/manual_add_single_word_tran', methods=['POST'])
def manual_add_single_word_tran():
    data = request.json
    keyword = data.get("keyword", "").strip().lower()  
    article_id = data.get("article_id", "").strip()
    content = data.get("content", "").strip()

    if not keyword or not article_id or article_id == "unknown":
        return jsonify({"success": False, "error": "Thiếu hoặc không hợp lệ: từ khóa hoặc article_id"})

    try:
        article = DiacriticSentence.query.filter_by(article_id=article_id).first()
        if not article:
            return jsonify({"success": False, "error": "Không tìm thấy bài viết với article_id này."})

        full_text = content.lower()
        frequency = len(re.findall(r'\b' + re.escape(keyword) + r'\b', full_text))

        existing = VietnameseDictionary.query.filter_by(keyword=keyword, article_id=article_id).first()
        if existing:
            return jsonify({"success": False, "error": "Từ này đã tồn tại."})

        _, part_of_speech = get_word_type(keyword)

        english_equivalent = translate_vi_to_en(keyword)

        definition = get_vi_meaning(keyword)

        new_word = VietnameseDictionary(
            article_id=article_id,
            keyword=keyword,
            part_of_speech=part_of_speech,
            english_equivalent=english_equivalent,
            definition=definition,
            frequency=frequency if frequency > 0 else 1  
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
        dict_words = VietnameseDictionary.query.with_entities(VietnameseDictionary.keyword).all()
        dict_keywords = [kw[0].lower() for kw in dict_words]

        dict_keywords.sort(key=lambda x: len(x), reverse=True)

        tokens = []
        processed_positions = set() 

        for keyword in dict_keywords:
            pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
            for match in pattern.finditer(content_lower):
                start, end = match.start(), match.end()
                if not any(start <= pos < end for pos in processed_positions):
                    tokens.append({
                        "token": match.group(),
                        "start": start,
                        "end": end
                    })
                    processed_positions.update(range(start, end))

        return jsonify({"success": True, "tokens": tokens})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/add-entity')
def add_entity():
    title = unquote(request.args.get('title', 'No Title'))
    content = unquote(request.args.get('content', 'No Content Available'))
    
    # Lấy article_id từ cơ sở dữ liệu dựa trên title
    article = Article.query.filter_by(title=title).first()
    article_id = article.article_id if article else 'unknown'

    return render_template(
        'add_entity.html',
        title=title,
        content=content,
        article_id=article_id
    )

@app.route('/add-entity_tran')
def add_entity_tran():
    content = unquote(request.args.get('content', 'No Content Available'))
    
    article = DiacriticSentence.query.filter_by(content=content).first()
    article_id = article.article_id if article else 'unknown'

    return render_template(
        'add_entity_tran.html',
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
        article = Article.query.filter_by(article_id=article_id).first()
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

@app.route('/api/manual_add_entity_tran', methods=['POST'])
def manual_add_entity_tran():
    data = request.json
    entity = data.get("entity", "").strip().lower()  
    entity_alias = data.get("entity_alias", "").strip().lower() 
    article_id = data.get("article_id", "").strip()
    content = data.get("content", "").strip()

    if not entity or not article_id or article_id == "unknown":
        return jsonify({"success": False, "error": "Thiếu hoặc không hợp lệ: thực thể hoặc article_id"})

    try:
        article = DiacriticSentence.query.filter_by(article_id=article_id).first()
        if not article:
            return jsonify({"success": False, "error": "Không tìm thấy bài viết với article_id này."})

        full_text = content.lower()
        entity_frequency = len(re.findall(r'\b' + re.escape(entity) + r'\b', full_text))
        entity_alias_frequency = len(re.findall(r'\b' + re.escape(entity_alias) + r'\b', full_text)) if entity_alias else 0

        existing_word = VietnameseDictionary.query.filter_by(keyword=entity, article_id=article_id).first()
        if existing_word:
            return jsonify({"success": False, "error": "Thực thể này đã tồn tại trong từ điển."})

        _, entity_pos = get_word_type(entity)
        entity_english = translate_vi_to_en(entity)
        entity_definition = get_vi_meaning(entity)

        new_word = VietnameseDictionary(
            article_id=article_id,
            keyword=entity,
            part_of_speech=entity_pos,
            english_equivalent=entity_english,
            definition=entity_definition,
            frequency=entity_frequency if entity_frequency > 0 else 1
        )
        db.session.add(new_word)

        if entity_alias:
            existing_alias_word = VietnameseDictionary.query.filter_by(keyword=entity_alias, article_id=article_id).first()
            if existing_alias_word:
                return jsonify({"success": False, "error": "Bí danh thực thể này đã tồn tại trong từ điển."})

            _, alias_pos = get_word_type(entity_alias)
            alias_english = translate_vi_to_en(entity_alias)
            alias_definition = get_vi_meaning(entity_alias)

            new_alias_word = VietnameseDictionary(
                article_id=article_id,
                keyword=entity_alias,
                part_of_speech=alias_pos,
                english_equivalent=alias_english,
                definition=alias_definition,
                frequency=entity_alias_frequency if entity_alias_frequency > 0 else 1
            )
            db.session.add(new_alias_word)

        existing_entity = Entity.query.filter_by(entity=entity, article_id=article_id).first()
        if not existing_entity:
            new_entity = Entity(
                article_id=article_id,
                entity=entity,
                entity_type="MISC"  
            )
            db.session.add(new_entity)

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


import tensorflow_datasets as tfds
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle

# --- Encode functions ---
def encode(ipt, opt):
    ipt = [tokenizer_ipt.vocab_size] + tokenizer_ipt.encode(
        ipt.numpy()) + [tokenizer_ipt.vocab_size+1]
    opt = [tokenizer_opt.vocab_size] + tokenizer_opt.encode(
        opt.numpy()) + [tokenizer_opt.vocab_size+1]
    return ipt, opt

def tf_encode(ipt, opt):
    result_ipt, result_opt = tf.py_function(encode, [ipt, opt], [tf.int64, tf.int64])
    result_ipt.set_shape([None])
    result_opt.set_shape([None])
    return result_ipt, result_opt

# --- Positional encoding functions ---
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                           np.arange(d_model)[np.newaxis, :],
                           d_model) # shape (position, d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32) # shape: (position, d_model)

# --- Masking functions ---
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

# --- Attention mechanisms ---
def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    dk = tf.cast(tf.shape(k)[-1], tf.float32) # depth
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    return output, attention_weights

def print_out(q, k, v):
    temp_out, temp_attn = scaled_dot_product_attention(
        q, k, v, None)
    print('Attention weights are:')
    print(temp_attn)
    print('Output is:')
    print(temp_out)

# --- MultiHeadAttention Layer ---
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, 
                                    (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        return output, attention_weights

# --- Feed Forward Network ---
def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

# --- Encoder Layer ---
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
    
    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        return out2

# --- Decoder Layer ---
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
    
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
        return out3, attn_weights_block1, attn_weights_block2

# --- Encoder ---
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                                self.d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                          for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
    
    def call(self, inputs, training=None, mask=None):
        seq_len = tf.shape(inputs)[1]
        x = self.embedding(inputs)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training, mask=mask)
        return x  # (batch_size, input_seq_len, d_model)

# --- Decoder ---
class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                          for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
    
    def call(self, inputs, enc_output, training=None, look_ahead_mask=None, padding_mask=None):
        seq_len = tf.shape(inputs)[1]
        attention_weights = {}
        x = self.embedding(inputs)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training=training,
                                            look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
        return x, attention_weights

# --- Transformer Model ---
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
                 target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                              input_vocab_size, pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                              target_vocab_size, pe_target, rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    
    def call(self, inp, tar, training=None, enc_padding_mask=None, 
             look_ahead_mask=None, dec_padding_mask=None):
        enc_output = self.encoder(inp, training=training, mask=enc_padding_mask)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training=training, look_ahead_mask=look_ahead_mask, padding_mask=dec_padding_mask)
        final_output = self.final_layer(dec_output)
        return final_output, attention_weights

# --- Pickle utilities ---
def _save_pickle(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def _load_pickle(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj

# --- Load tokenizers ---
tokenizer_ipt = _load_pickle('tokenizer/tokenizer_ipt.pkl')
tokenizer_opt = _load_pickle('tokenizer/tokenizer_opt.pkl')

# --- Model parameters ---
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
input_vocab_size = tokenizer_ipt.vocab_size + 2
target_vocab_size = tokenizer_opt.vocab_size + 2
dropout_rate = 0.1
learning_rate = 0.01

# --- Learning rate schedule ---
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                    epsilon=1e-9)

# --- Loss function ---
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

# --- Metrics ---
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

# --- Create masks ---
def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return enc_padding_mask, combined_mask, dec_padding_mask

# --- Plot attention weights ---
def plot_attention_weights(attention, sentence, result, layer):
    fig = plt.figure(figsize=(16, 8))
    sentence = tokenizer_ipt.encode(sentence)
    attention = tf.squeeze(attention[layer], axis=0)
    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head+1)
        ax.matshow(attention[head][:-1, :], cmap='viridis')
        fontdict = {'fontsize': 10}
        ax.set_xticks(range(len(sentence)+2))
        ax.set_yticks(range(len(result)))
        ax.set_ylim(len(result)-1.5, -0.5)
        ax.set_xticklabels(
            ['<start>']+[tokenizer_ipt.decode([i]) for i in sentence]+['<end>'], 
            fontdict=fontdict, rotation=90)
        ax.set_yticklabels([tokenizer_opt.decode([i]) for i in result 
                           if i < tokenizer_opt.vocab_size], 
                          fontdict=fontdict)
        ax.set_xlabel('Head {}'.format(head+1))
    plt.tight_layout()
    plt.show()

# --- Initialize Transformer and Checkpoint ---
transformer = Transformer(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
                         input_vocab_size=input_vocab_size, target_vocab_size=target_vocab_size, 
                         pe_input=input_vocab_size, 
                         pe_target=target_vocab_size,
                         rate=dropout_rate)

checkpoint_path = "./checkpoints/train_500k"
ckpt = tf.train.Checkpoint(transformer=transformer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)

# --- Evaluation functions ---
def evaluate(inp_sentence):
    inp_sentence = inp_sentence.lower()
    start_token = [input_vocab_size - 2]
    end_token = [input_vocab_size - 1]
    inp_sentence = start_token + tokenizer_ipt.encode(inp_sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)
    decoder_input = [target_vocab_size - 2]
    output = tf.expand_dims(decoder_input, 0)
    for i in range(40):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)
        predictions, attention_weights = transformer(encoder_input, 
                                                    output,
                                                    training=False,
                                                    enc_padding_mask=enc_padding_mask,
                                                    look_ahead_mask=combined_mask,
                                                    dec_padding_mask=dec_padding_mask)
        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        if predicted_id == target_vocab_size - 1:
            return tf.squeeze(output, axis=0), attention_weights
        output = tf.concat([output, predicted_id], axis=-1)
    return tf.squeeze(output, axis=0), attention_weights

def add_diacritic(sentence, plot=''):
    result, attention_weights = evaluate(sentence)
    predicted_sentence = tokenizer_opt.decode([i for i in result 
                                             if i < target_vocab_size - 2])
    print('Predicted translation: {}'.format(predicted_sentence))
    if plot:
        plot_attention_weights(attention_weights, sentence, result, plot)


@app.route('/api/add-diacritics', methods=['POST'])
def api_add_diacritics():
    data = request.get_json()
    input_text = data.get("undiacritic", "")
    try:
        result, _ = evaluate(input_text)
        predicted_sentence = tokenizer_opt.decode([i for i in result if i < target_vocab_size - 2])
        return jsonify({"success": True, "diacritic_content": predicted_sentence})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
    
@app.route('/api/save_diacritic_sentence', methods=['POST'])
def save_diacritic_sentence():
    data = request.json
    content = data.get("content", "").strip()
    undiacritic = data.get("undiacritic", "").strip()  

    if not undiacritic or not content:
        return jsonify({"success": False, "error": "Thiếu nội dung"})

    try:
        # Tìm bản ghi đã có với content
        sentence = DiacriticSentence.query.filter_by(undiacritic=undiacritic).first()
        if not sentence:
            return jsonify({"success": False, "error": "Không tìm thấy bản ghi để cập nhật."})

        # Cập nhật cột undiacritic
        sentence.content = content
        db.session.commit()

        return jsonify({"success": True})
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)