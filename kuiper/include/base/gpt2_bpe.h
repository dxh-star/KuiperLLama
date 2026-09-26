
#ifndef KUIPER_INCLUDE_BASE_GPT2_BPE_H_
#define KUIPER_INCLUDE_BASE_GPT2_BPE_H_
// GPT-2 byte-level BPE（自研，无 absl/re2/nlohmann 依赖；配合 export_gpt2_tokenizer.py）
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <limits>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
namespace base {
namespace gpt2bpe {

inline std::string utf8_encode(int cp) {
  std::string s;
  if (cp < 0x80) s += static_cast<char>(cp);
  else if (cp < 0x800) { s += static_cast<char>(0xC0|(cp>>6)); s += static_cast<char>(0x80|(cp&0x3F)); }
  else { s += static_cast<char>(0xE0|(cp>>12)); s += static_cast<char>(0x80|((cp>>6)&0x3F)); s += static_cast<char>(0x80|(cp&0x3F)); }
  return s;
}
inline char32_t utf8_decode_one(const std::string& s) {
  if (s.empty()) return 0;
  const unsigned char c0 = static_cast<unsigned char>(s[0]);
  if ((c0&0x80)==0) return c0;
  if ((c0&0xE0)==0xC0 && s.size()>=2) return ((c0&0x1F)<<6)|(static_cast<unsigned char>(s[1])&0x3F);
  if ((c0&0xF0)==0xE0 && s.size()>=3) return ((c0&0x0F)<<12)|((static_cast<unsigned char>(s[1])&0x3F)<<6)|(static_cast<unsigned char>(s[2])&0x3F);
  if ((c0&0xF8)==0xF0 && s.size()>=4) return ((c0&0x07)<<18)|((static_cast<unsigned char>(s[1])&0x3F)<<12)|((static_cast<unsigned char>(s[2])&0x3F)<<6)|(static_cast<unsigned char>(s[3])&0x3F);
  return c0;
}
inline std::u32string utf8_to_u32(const std::string& s) {
  std::u32string out; size_t i=0;
  while (i<s.size()) {
    const unsigned char c0=static_cast<unsigned char>(s[i]); size_t len=1;
    if ((c0&0xE0)==0xC0) len=2; else if ((c0&0xF0)==0xE0) len=3; else if ((c0&0xF8)==0xF0) len=4;
    out.push_back(utf8_decode_one(s.substr(i,len))); i+=len;
  }
  return out;
}
inline const std::array<std::string,256>& byte_encoder_table() {
  static const std::array<std::string,256> table = [] {
    std::array<std::string,256> t{}; std::vector<int> bs,cs;
    for (int b='!'; b<='~'; ++b) bs.push_back(b);
    for (int b=0xA1; b<=0xAC; ++b) bs.push_back(b);
    for (int b=0xAE; b<=0xFF; ++b) bs.push_back(b);
    cs=bs; int n=0;
    for (int b=0; b<256; ++b) if (std::find(bs.begin(),bs.end(),b)==bs.end()) { bs.push_back(b); cs.push_back(256+n); ++n; }
    for (size_t i=0;i<bs.size();++i) t[bs[i]]=utf8_encode(cs[i]);
    return t;
  }();
  return table;
}

class JsonValue {
 public:
  enum class Type { kNull,kBool,kInt,kString,kArray,kObject };
  Type type=Type::kNull; int64_t num=0; std::string str;
  std::vector<JsonValue> arr; std::vector<std::pair<std::string,JsonValue>> obj;
  const JsonValue* find(const std::string& key) const {
    for (const auto& kv:obj) if (kv.first==key) return &kv.second;
    return nullptr;
  }
};
class JsonParser {
 public:
  explicit JsonParser(const std::string& t):text_(t){}
  bool parse(JsonValue& out){ skip_ws(); return parse_value(out); }
 private:
  const std::string& text_; size_t pos_=0;
  void skip_ws(){ while(pos_<text_.size()&&(text_[pos_]==' '||text_[pos_]=='\t'||text_[pos_]=='\n'||text_[pos_]=='\r')) ++pos_; }
  bool parse_value(JsonValue& out){
    skip_ws(); if(pos_>=text_.size()) return false;
    char c=text_[pos_];
    if(c=='{') return parse_object(out);
    if(c=='[') return parse_array(out);
    if(c=='"'){ out.type=JsonValue::Type::kString; return parse_string(out.str); }
    if(c=='t'||c=='f'||c=='n'){
      const char* lit=(c=='t')?"true":(c=='f')?"false":"null"; size_t len=(c=='t')?4:(c=='f')?5:4;
      if(text_.compare(pos_,len,lit)!=0) return false;
      out.type=(c=='n')?JsonValue::Type::kNull:JsonValue::Type::kBool; out.num=(c=='t')?1:0; pos_+=len; return true;
    }
    return parse_number(out);
  }
  bool parse_number(JsonValue& out){
    size_t st=pos_;
    if(pos_<text_.size()&&(text_[pos_]=='-'||text_[pos_]=='+')) ++pos_;
    while(pos_<text_.size()&&text_[pos_]>='0'&&text_[pos_]<='9') ++pos_;
    while(pos_<text_.size()&&(text_[pos_]=='.'||text_[pos_]=='e'||text_[pos_]=='E'||text_[pos_]=='-'||text_[pos_]=='+')) ++pos_;
    if(pos_==st) return false;
    out.type=JsonValue::Type::kInt; out.num=std::strtoll(text_.substr(st,pos_-st).c_str(),nullptr,10); return true;
  }
  bool parse_string(std::string& out){
    if(pos_>=text_.size()||text_[pos_]!='"') return false;
    ++pos_; out.clear();
    while(pos_<text_.size()){
      char c=text_[pos_++];
      if(c=='"') return true;
      if(c!='\\'){ out+=c; continue; }
      if(pos_>=text_.size()) return false;
      char e=text_[pos_++];
      switch(e){
        case '"':out+='"';break; case '\\':out+='\\';break; case '/':out+='/';break;
        case 'b':out+='\b';break; case 'f':out+='\f';break; case 'n':out+='\n';break;
        case 'r':out+='\r';break; case 't':out+='\t';break;
        case 'u':{
          if(pos_+4>text_.size()) return false;
          int cp=static_cast<int>(std::strtol(text_.substr(pos_,4).c_str(),nullptr,16)); pos_+=4;
          if(cp>=0xD800&&cp<=0xDBFF&&pos_+6<=text_.size()&&text_[pos_]=='\\'&&text_[pos_+1]=='u'){
            int lo=static_cast<int>(std::strtol(text_.substr(pos_+2,4).c_str(),nullptr,16));
            if(lo>=0xDC00&&lo<=0xDFFF){ cp=0x10000+((cp-0xD800)<<10)+(lo-0xDC00); pos_+=6; }
          }
          out+=utf8_encode(cp); break;
        }
        default:return false;
      }
    }
    return false;
  }
  bool parse_array(JsonValue& out){
    out.type=JsonValue::Type::kArray; ++pos_; skip_ws();
    if(pos_<text_.size()&&text_[pos_]==']'){ ++pos_; return true; }
    while(true){
      JsonValue item; if(!parse_value(item)) return false;
      out.arr.push_back(std::move(item)); skip_ws();
      if(pos_>=text_.size()) return false;
      if(text_[pos_]==','){ ++pos_; continue; }
      if(text_[pos_]==']'){ ++pos_; return true; }
      return false;
    }
  }
  bool parse_object(JsonValue& out){
    out.type=JsonValue::Type::kObject; ++pos_; skip_ws();
    if(pos_<text_.size()&&text_[pos_]=='}'){ ++pos_; return true; }
    while(true){
      skip_ws(); std::string key;
      if(!parse_string(key)) return false;
      skip_ws(); if(pos_>=text_.size()||text_[pos_]!=':') return false; ++pos_;
      JsonValue val; if(!parse_value(val)) return false;
      out.obj.emplace_back(std::move(key),std::move(val)); skip_ws();
      if(pos_>=text_.size()) return false;
      if(text_[pos_]==','){ ++pos_; continue; }
      if(text_[pos_]=='}'){ ++pos_; return true; }
      return false;
    }
  }
};

class GPT2Bpe {
 public:
  static const char* pattern() {
    return "'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z\\x80-\\xFF]+| ?[0-9]+| "
           "?[^ \\t\\r\\n\\f\\vA-Za-z0-9\\x80-\\xFF]+|\\s+(?!\\S)|\\s+";
  }
  bool load(const std::string& json_path) {
    std::ifstream ifs(json_path,std::ios::binary);
    if(!ifs.good()) return false;
    std::stringstream ss; ss<<ifs.rdbuf(); const std::string text=ss.str();
    if(text.empty()) return false;
    JsonValue root; JsonParser parser(text);
    if(!parser.parse(root)) return false;
    const JsonValue* model=root.find("model");
    if(model==nullptr) return false;
    const JsonValue* vocab=model->find("vocab");
    const JsonValue* merges=model->find("merges");
    if(vocab==nullptr||vocab->type!=JsonValue::Type::kObject) return false;
    for(const auto& kv:vocab->obj){
      ranks_.emplace(utf8_to_u32(kv.first),static_cast<int>(kv.second.num));
      decoder_.emplace(static_cast<int>(kv.second.num),kv.first);
    }
    (void)merges;
    const JsonValue* added=root.find("added_tokens");
    if(added!=nullptr&&added->type==JsonValue::Type::kArray){
      for(const auto& t:added->arr){
        const JsonValue* content=t.find("content"); const JsonValue* id=t.find("id");
        if(content!=nullptr&&id!=nullptr&&!content->str.empty()) special_decoder_.emplace(static_cast<int>(id->num),content->str);
      }
    }
    regex_=std::regex(std::string("(")+pattern()+")",std::regex::ECMAScript);
    return !ranks_.empty();
  }
  int32_t vocab_size() const { return static_cast<int32_t>(decoder_.size()); }
  std::vector<int32_t> encode(const std::string& text) const {
    std::vector<int32_t> ids;
    std::sregex_iterator it(text.begin(),text.end(),regex_), end;
    for(;it!=end;++it){
      const std::u32string piece=byte_encode(it->str());
      if(piece.empty()) continue;
      auto merged=byte_pair_merge(piece);
      ids.insert(ids.end(),merged.begin(),merged.end());
    }
    return ids;
  }
  std::string decode(const std::vector<int32_t>& ids) const {
    std::string encoded;
    for(int32_t id:ids){
      auto it=decoder_.find(id);
      if(it!=decoder_.end()){ encoded+=it->second; continue; }
      auto sit=special_decoder_.find(id);
      if(sit!=special_decoder_.end()) encoded+=sit->second;
    }
    return byte_decode(encoded);
  }
  std::string decode_token(int32_t id) const {
    auto it=decoder_.find(id);
    if(it==decoder_.end()) return std::string();
    return byte_decode(it->second);
  }
 private:
  std::vector<int32_t> byte_pair_merge(const std::u32string& piece) const {
    std::vector<std::pair<int,int>> parts;
    parts.reserve(piece.size()+1);
    for(size_t i=0;i<=piece.size();++i) parts.emplace_back(static_cast<int>(i),std::numeric_limits<int>::max());
    auto get_rank=[&](const std::vector<std::pair<int,int>>& p,int idx,int skip)->int{
      if(idx+skip+2<static_cast<int>(p.size())){
        const int start=p[idx].first, stop=p[idx+skip+2].first;
        auto it=ranks_.find(piece.substr(start,stop-start));
        if(it!=ranks_.end()) return it->second;
      }
      return std::numeric_limits<int>::max();
    };
    for(int i=0;i+2<static_cast<int>(parts.size());++i) parts[i].second=get_rank(parts,i,0);
    while(parts.size()>1){
      int min_rank=std::numeric_limits<int>::max(),min_idx=-1;
      for(int i=0;i+1<static_cast<int>(parts.size());++i) if(parts[i].second<min_rank){min_rank=parts[i].second;min_idx=i;}
      if(min_idx==-1) break;
      parts[min_idx].second=std::numeric_limits<int>::max();
      parts.erase(parts.begin()+min_idx+1);
      if(min_idx>0) parts[min_idx-1].second=get_rank(parts,min_idx-1,0);
      if(min_idx<static_cast<int>(parts.size())-1) parts[min_idx].second=get_rank(parts,min_idx,0);
    }
    std::vector<int32_t> result;
    result.reserve(parts.size()-1);
    for(size_t i=0;i+1<parts.size();++i){
      const int start=parts[i].first, stop=parts[i+1].first;
      auto it=ranks_.find(piece.substr(start,stop-start));
      if(it==ranks_.end()) continue;
      result.push_back(it->second);
    }
    return result;
  }
  std::u32string byte_encode(const std::string& text) const {
    const auto& table=byte_encoder_table();
    std::u32string out; out.reserve(text.size());
    for(unsigned char c:text) out.push_back(utf8_decode_one(table[c]));
    return out;
  }
  std::string byte_decode(const std::string& encoded) const {
    std::string out; size_t i=0;
    while(i<encoded.size()){
      unsigned char c=static_cast<unsigned char>(encoded[i]); size_t len=1;
      if((c&0xE0)==0xC0) len=2; else if((c&0xF0)==0xE0) len=3; else if((c&0xF8)==0xF0) len=4;
      std::string ch=encoded.substr(i,len);
      auto it=reverse_.find(ch);
      if(it!=reverse_.end()) out+=static_cast<char>(it->second); else out+=ch;
      i+=len;
    }
    return out;
  }
  std::unordered_map<std::u32string,int> ranks_;
  std::unordered_map<int,std::string> decoder_;
  std::unordered_map<int,std::string> special_decoder_;
  std::unordered_map<std::string,uint8_t> reverse_;
  std::regex regex_;
  void init_reverse(){ const auto& table=byte_encoder_table(); for(int b=0;b<256;++b) reverse_[table[b]]=static_cast<uint8_t>(b); }
 public:
  GPT2Bpe(){ init_reverse(); }
};

}  // namespace gpt2bpe
}  // namespace base
#endif
