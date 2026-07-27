from pathlib import Path
import os

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent


def ensure_env_loaded() -> None:
	"""Load environment variables from the workspace .env file once."""
	load_dotenv(BASE_DIR / ".env", override=False)


ensure_env_loaded()

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = DATA_DIR / "output"
DB_PATH = BASE_DIR / "db.db"

DEFAULT_START_YEAR = 2015
DEFAULT_END_YEAR = 2025

# LLM pipeline directories and metadata
STAGING_DIR = DATA_DIR / "staging"
MARKDOWN_DIR = DATA_DIR / "markdown"
BCTC_MARKDOWN_DIR = DATA_DIR / "markdown_bctc"
LOGS_DIR = DATA_DIR / "logs"
ENV_JSON_PATH = BASE_DIR / "env.json"

# Embedding settings
EMBEDDING_MODEL: str = "text-embedding-3-small"
EMBEDDING_DIMENSIONS: int = 1536
EMBEDDING_CHUNK_SIZE: int = 512
EMBEDDING_CHUNK_OVERLAP: int = 128
EMBEDDING_BATCH_SIZE: int = 512

# Inference settings
INFERENCE_MODEL: str = "gpt-4.1-mini"
INFERENCE_TOP_K: int = 20
INFERENCE_TEMPERATURE: float = float(os.getenv("INFERENCE_TEMPERATURE", "0"))
HYDE2_ENABLED: bool = os.getenv("HYDE2_ENABLED", "1").strip().lower() in {
	"1",
	"true",
	"yes",
	"on",
}
HYDE2_MODEL: str = os.getenv("HYDE2_MODEL", INFERENCE_MODEL)
HYDE2_SYNTHETIC_DOC_COUNT: int = int(os.getenv("HYDE2_SYNTHETIC_DOC_COUNT", "2"))

# Phase 1 hybrid retrieval scoring weights for governance RAG.
INFERENCE_RETRIEVAL_ALPHA: float = float(
	os.getenv("INFERENCE_RETRIEVAL_ALPHA", "1.0")
)
INFERENCE_RETRIEVAL_BETA: float = float(
	os.getenv("INFERENCE_RETRIEVAL_BETA", "0.12")
)
INFERENCE_RETRIEVAL_GAMMA: float = float(
	os.getenv("INFERENCE_RETRIEVAL_GAMMA", "0.08")
)
INFERENCE_RETRIEVAL_CANDIDATE_MULTIPLIER: int = int(
	os.getenv("INFERENCE_RETRIEVAL_CANDIDATE_MULTIPLIER", "4")
)

# Environmental Disclosure Checklist
CHECKLIST_ITEMS: list[dict[str, str]] = [
	{
		"code": "CC1",
		"group": "Climate Change",
		"item": "Assessment of climate-related risks and opportunities",
		"description": "Đánh giá các rủi ro (các quy định, tác động vật lý hoặc các tác động chung) liên quan đến biến đổi khí hậu và các hành động đã hoặc sẽ thực hiện để quản lý rủi ro.",
	},
	{
		"code": "CC2",
		"group": "Climate Change",
		"item": "Financial implications of climate change",
		"description": "Đánh giá các tác động tài chính hiện tại (và tương lai), tác động kinh doanh và cơ hội của biến đổi khí hậu.",
	},
	{
		"code": "GHG1",
		"group": "GHG Emissions",
		"item": "Methodology for GHG emission calculation",
		"description": "Mô tả các phương pháp sử dụng để tính toán khí thải nhà kính.",
	},
	{
		"code": "GHG2",
		"group": "GHG Emissions",
		"item": "External verification of GHG emissions",
		"description": "Có sự xác nhận bởi yếu tố bên ngoài về lượng phát thải khí nhà kính hay không? Nếu có, bởi ai và trên cơ sở gì?",
	},
	{
		"code": "GHG3",
		"group": "GHG Emissions",
		"item": "Total GHG emissions disclosed",
		"description": "Lượng phát thải khí nhà kính tính bằng đơn vị MtCO2e (hệ mét tấn CO2 thải ra).",
	},
	{
		"code": "GHG4",
		"group": "GHG Emissions",
		"item": "Disclosure of emissions by scope (Scope 1, 2, 3)",
		"description": "Việc công bố liên quan đến Phạm vi 1, Phạm vi 2, Phạm vi 3 và liên quan trực tiếp đến phát thải khí nhà kính.",
	},
	{
		"code": "GHG5",
		"group": "GHG Emissions",
		"item": "Disclosure of emissions by source",
		"description": "Công bố phát thải nhà kính dựa trên nguồn phát thải nào?",
	},
	{
		"code": "GHG6",
		"group": "GHG Emissions",
		"item": "Disclosure of emissions by facility or segment",
		"description": "Công bố phát thải khí nhà kính dựa trên cơ sở vật chất hoặc cấp độ nào?",
	},
	{
		"code": "GHG7",
		"group": "GHG Emissions",
		"item": "Historical comparison of emissions over time",
		"description": "So sánh lượng phát thải khí nhà kính trong năm hiện tại và năm trước.",
	},
	{
		"code": "EC1",
		"group": "Energy Consumption",
		"item": "Total energy consumed",
		"description": "Lượng năng lượng tiêu thụ.",
	},
	{
		"code": "EC2",
		"group": "Energy Consumption",
		"item": "Disclosure of energy consumption from renewable sources",
		"description": "Lượng năng lượng tiêu thụ có nguồn gốc từ nguồn năng lượng tái tạo.",
	},
	{
		"code": "EC3",
		"group": "Energy Consumption",
		"item": "Disclosure of energy consumption by type, facility, or segment",
		"description": "Tiết lộ dựa trên loại khí thải, cơ sở vật chất hoặc cấp độ nào?",
	},
	{
		"code": "RC1",
		"group": "GHG Reduction",
		"item": "Plans or strategies to reduce GHG emissions",
		"description": "Giải thích chi tiết về chiến lược và kế hoạch giảm thiểu phát thải khí nhà kính.",
	},
	{
		"code": "RC2",
		"group": "GHG Reduction",
		"item": "Specific targets for GHG emission reduction",
		"description": "Mục tiêu cụ thể về lượng giảm thiểu phát thải nhà kính.",
	},
	{
		"code": "RC3",
		"group": "GHG Reduction",
		"item": "Reductions achieved to date",
		"description": "Lượng giảm phát thải tối đa và các chi phí hoặc tiết kiệm liên quan đến giảm phát khí thải nhà kính tính đến thời điểm báo cáo.",
	},
	{
		"code": "RC4",
		"group": "GHG Reduction",
		"item": "Costs of future emissions factored into capital expenditure planning",
		"description": "Mức độ chi phí liên quan đến phát thải khí nhà kính trong tương lai vì chi phí này được bao gồm trong kế hoạch sử dụng vốn của công ty.",
	},
	{
		"code": "ACC1",
		"group": "GHG Accountability",
		"item": "Responsibility for climate change policy and action",
		"description": "Giải thích ai là Ủy ban hoặc Giám đốc chịu trách nhiệm về các chính sách liên quan đến biến đổi khí hậu.",
	},
	{
		"code": "ACC2",
		"group": "GHG Accountability",
		"item": "Consideration of climate-related goals in executive compensation",
		"description": "Giải thích cơ chế xem xét khi đạt được mục tiêu công ty liên quan đến biến đổi khí hậu bởi Ủy ban hoặc Hội đồng quản trị.",
	},
]

# More direct, broader EDC alternatives to reduce false negatives.
_CHECKLIST_ALT_DESCRIPTIONS: dict[str, str] = {
	"CC1": "Chấp nhận nếu có mô tả rủi ro/cơ hội khí hậu hoặc rủi ro môi trường liên quan (nắng nóng, ngập lụt, gián đoạn chuỗi cung ứng, chuyển dịch chính sách) và hành động ứng phó ở mức định tính.",
	"CC2": "Chấp nhận nếu có đề cập tác động tài chính hoặc kinh doanh do yếu tố khí hậu/môi trường, kể cả chỉ nêu xu hướng chi phí, đầu tư, doanh thu, hoặc rủi ro mà chưa có số liệu đầy đủ.",
	"GHG1": "Chấp nhận nếu có mô tả cách doanh nghiệp theo dõi/đo lường phát thải hoặc tiêu thụ năng lượng liên quan phát thải, dù chưa nêu chuẩn phương pháp chi tiết.",
	"GHG2": "Chấp nhận nếu có bất kỳ đề cập xác minh/đánh giá bên ngoài liên quan dữ liệu môi trường hoặc hệ thống quản lý môi trường, không bắt buộc xác minh riêng chỉ tiêu GHG.",
	"GHG3": "Chấp nhận nếu có công bố bất kỳ số liệu phát thải/carbon/CO2 tương đương hoặc chỉ số thay thế liên quan phát thải, kể cả chưa chuẩn hóa đơn vị MtCO2e.",
	"GHG4": "Chấp nhận nếu có tách nhóm phát thải theo loại/phạm vi/nguồn tương đương (trực tiếp, gián tiếp, điện năng, nhiên liệu) dù không ghi đúng nhãn Scope 1/2/3.",
	"GHG5": "Chấp nhận nếu có mô tả nguồn phát thải chính (điện, nhiên liệu, vận tải, sản xuất, logistics...) ở bất kỳ mức chi tiết nào.",
	"GHG6": "Chấp nhận nếu có phân tách phát thải hoặc chỉ số liên quan theo nhà máy, đơn vị, khu vực, mảng kinh doanh, hoặc bất kỳ phân khúc vận hành nào.",
	"GHG7": "Chấp nhận nếu có so sánh theo thời gian của phát thải hoặc chỉ số môi trường liên quan (năm trước/sau, xu hướng tăng giảm), kể cả chỉ nêu định tính.",
	"EC1": "Chấp nhận nếu có công bố tổng tiêu thụ năng lượng hoặc chỉ số đại diện (điện, nhiên liệu, nhiệt), không bắt buộc đầy đủ mọi loại năng lượng.",
	"EC2": "Chấp nhận nếu có đề cập sử dụng năng lượng tái tạo hoặc nguồn năng lượng sạch, kể cả mô tả chương trình/chủ trương mà chưa có số liệu chi tiết.",
	"EC3": "Chấp nhận nếu có bất kỳ phân tách tiêu thụ năng lượng theo loại năng lượng, đơn vị vận hành, cơ sở, hoặc phân khúc hoạt động.",
	"RC1": "Chấp nhận nếu có nêu kế hoạch/chương trình/sáng kiến giảm phát thải hoặc giảm tác động khí hậu ở mức định hướng, kể cả chưa có KPI định lượng.",
	"RC2": "Chấp nhận nếu có mục tiêu giảm phát thải hoặc mục tiêu môi trường tương đương (tiết kiệm năng lượng, giảm cường độ phát thải, net-zero) dù chưa đủ baseline hoặc mốc thời gian chi tiết.",
	"RC3": "Chấp nhận nếu có đề cập kết quả đã đạt được liên quan giảm phát thải/giảm tiêu thụ năng lượng/hiệu quả tài nguyên, kể cả chỉ nêu phần trăm hoặc mô tả kết quả định tính.",
	"RC4": "Chấp nhận nếu có đề cập yếu tố môi trường/phát thải được cân nhắc trong đầu tư, mua sắm, nâng cấp công nghệ, hoặc quyết định vốn, không bắt buộc lượng hóa chi phí carbon.",
	"ACC1": "Chấp nhận nếu có chỉ ra bộ phận/chức danh chịu trách nhiệm về môi trường, khí hậu, ESG hoặc phát triển bền vững, không bắt buộc nêu đúng cấp ủy ban/hội đồng.",
	"ACC2": "Chấp nhận nếu có liên hệ giữa mục tiêu môi trường/ESG và đánh giá hiệu quả, thưởng, KPI quản lý, hoặc cơ chế giám sát điều hành, kể cả gián tiếp.",
}

CHECKLIST_ITEMS_ALT: list[dict[str, str]] = [
	{
		**item,
		"code": f"{item['code']}_alt",
		"item": f"{item['item']} (Alternative)",
		"description": _CHECKLIST_ALT_DESCRIPTIONS.get(item["code"], item["description"]),
	}
	for item in CHECKLIST_ITEMS
]

# PROPER-VN items
PROPER_VN_STAGE1_ITEMS: list[dict[str, str]] = [
	{
		"code": "S1_VIOLATION",
		"group": "Stage 1: Regulatory Compliance",
		"item": "Serious environmental violations",
		"description": "Có bằng chứng hoặc đề cập đến vi phạm nghiêm trọng về môi trường, bị xử phạt hành chính nặng, gây ô nhiễm, hoặc bị đình chỉ hoạt động vì lý do môi trường.",
		"score_if_true": "black",
	},
	{
		"code": "S1_MINOR_NC",
		"group": "Stage 1: Regulatory Compliance",
		"item": "Minor regulatory non-compliance",
		"description": "Có đề cập đến vi phạm nhỏ về quy định môi trường, chưa tuân thủ đầy đủ các tiêu chuẩn phát thải, xử lý chất thải hoặc quản lý môi trường theo quy định hiện hành.",
		"score_if_true": "red",
	},
	{
		"code": "S1_COMPLIANCE",
		"group": "Stage 1: Regulatory Compliance",
		"item": "Stated regulatory compliance",
		"description": "Công ty tuyên bố tuân thủ các quy định về môi trường, có giấy phép môi trường, đánh giá tác động môi trường (DTM/EIA), hoặc báo cáo tuân thủ quy chuẩn kỹ thuật môi trường.",
		"score_if_true": "compliant",
	},
]

PROPER_VN_STAGE2_ITEMS: list[dict[str, str]] = [
	{
		"code": "S2_ISO14001",
		"group": "Stage 2: Beyond-Compliance",
		"item": "ISO 14001 certification",
		"description": "Công ty có chứng nhận ISO 14001 hoặc hệ thống quản lý môi trường tương đương (EMS). Bao gồm việc đề cập đến việc đạt, duy trì hoặc tái chứng nhận ISO 14001.",
	},
	{
		"code": "S2_CARBON_DISC",
		"group": "Stage 2: Beyond-Compliance",
		"item": "Carbon emission disclosure",
		"description": "Công ty công bố lượng phát thải carbon hoặc khí nhà kính (GHG) cụ thể, bao gồm Scope 1, Scope 2, hoặc Scope 3, hoặc tổng phát thải tính bằng tấn CO2 tương đương.",
	},
	{
		"code": "S2_REDUCTION",
		"group": "Stage 2: Beyond-Compliance",
		"item": "Emission reduction targets or achievements",
		"description": "Công ty có đặt mục tiêu giảm phát thải khí nhà kính cụ thể (ví dụ: giảm X% vào năm Y), hoặc đã đạt được kết quả giảm phát thải so với năm gốc, hoặc cam kết Net Zero/Carbon Neutral.",
	},
	{
		"code": "S2_EFFICIENCY",
		"group": "Stage 2: Beyond-Compliance",
		"item": "Energy or resource efficiency initiatives",
		"description": "Công ty thực hiện các sáng kiến tiết kiệm năng lượng, sử dụng năng lượng tái tạo, tái chế tài nguyên, giảm tiêu thụ nước, hoặc các chương trình cải thiện hiệu suất sử dụng tài nguyên.",
	},
]

PROPER_VN_ALL_ITEMS: list[dict[str, str]] = (
	PROPER_VN_STAGE1_ITEMS + PROPER_VN_STAGE2_ITEMS
)

# Governance extraction items
GOVERNANCE_EXTRACTION_ITEMS: list[dict[str, str]] = [
	{
		"code": "GOV_SHAREHOLDERS",
		"group": "Shareholders",
		"item": "Major shareholders and ownership structure",
		"description": (
			"Trích xuất danh sách cổ đông được đề cập trong báo cáo. "
			"Liệt kê cổ đông lớn (sở hữu từ 5% trở lên) và cổ đông khác nếu có. "
			"Với mỗi cổ đông: tên, số cổ phần sở hữu, tỷ lệ sở hữu (%), "
			"loại cổ đông (cá nhân/tổ chức/nhà nước/nước ngoài). "
			"Tổng hợp: tổng số cổ phiếu đang lưu hành, cổ phiếu quỹ, "
			"tỷ lệ sở hữu nước ngoài, tỷ lệ sở hữu nhà nước, "
			"số cổ phần do tổ chức nắm giữ, số cổ phần do cá nhân nắm giữ."
		),
		"json_template": """{
	"value": {
		"total_outstanding_shares": null,
		"total_treasury_shares": null,
		"foreign_ownership_pct": null,
		"state_ownership_pct": null,
		"institutional_shares": null,
		"individual_shares": null
	},
	"details": [
		{
			"name": null,
			"shares": null,
			"ownership_pct": null,
			"type": null,
			"notes": null
		}
	],
	"reason": null
}""",
	},
	{
		"code": "GOV_DIRECTORY",
		"group": "Board of Directors",
		"item": "Board of directors full member list",
		"description": (
			"Trích xuất danh sách thành viên Hội đồng quản trị (HĐQT). "
			"Với mỗi người: họ tên, chức vụ HĐQT, giới tính, ngày sinh, "
			"ngày bổ nhiệm, nhiệm kỳ, trình độ học vấn, số cổ phần, tỷ lệ sở hữu, "
			"thành viên độc lập hay không. "
			"Nếu là thành viên độc lập thì ghi rõ lý do/chứng cứ xác định độc lập. "
			"Không trích xuất Ban Điều hành trong item này."
		),
		"json_template": """{
	"value": {
		"total_members": null,
		"women_count": null,
		"men_count": null,
		"foreign_count": null,
		"independent_count": null,
		"chair_name": null
	},
	"details": [
		{
			"name": null,
			"position": null,
			"gender": null,
			"date_of_birth": null,
			"appointment_date": null,
			"term_end": null,
			"education": null,
			"shares_owned": null,
			"ownership_pct": null,
			"is_independent": null,
			"independent_reason": null,
			"notes": null
		}
	],
	"reason": null
}""",
	},
	{
		"code": "GOV_EXECUTIVE",
		"group": "Executive Management",
		"item": "Executive management member list",
		"description": (
			"Trích xuất danh sách Ban Điều hành và các chức danh điều hành "
			"(Tổng Giám đốc/CEO, Phó Tổng Giám đốc, Kế toán trưởng, chức danh điều hành khác). "
			"Với mỗi người: họ tên, chức vụ điều hành, giới tính, ngày sinh, "
			"ngày bổ nhiệm, trình độ học vấn, có đồng thời là thành viên HĐQT hay không. "
			"Nếu là vai trò điều hành thì ghi rõ lý do/chứng cứ từ chức danh."
		),
		"json_template": """{
	"value": {
		"total_members": null,
		"women_count": null,
		"men_count": null,
		"ceo_name": null,
		"chief_accountant_name": null,
		"board_member_count": null
	},
	"details": [
		{
			"name": null,
			"position": null,
			"gender": null,
			"date_of_birth": null,
			"appointment_date": null,
			"education": null,
			"is_executive": null,
			"executive_reason": null,
			"is_board_member": null,
			"notes": null
		}
	],
	"reason": null
}""",
	},
	{
		"code": "GOV_AUDIT",
		"group": "Audit",
		"item": "Audit firm and audit committee details",
		"description": (
			"Trích xuất thông tin kiểm toán độc lập của công ty: "
			"tên công ty kiểm toán độc lập, ý kiến kiểm toán, "
			"kiểm toán viên ký báo cáo. "
			"Không trích xuất thành viên Ban Kiểm soát hoặc thành viên Ủy ban Kiểm toán nội bộ."
		),
		"json_template": """{
	"value": {
		"external_audit_firm": null,
		"external_audit_firm_en": null,
		"audit_opinion": null,
		"signing_auditor_names": []
	},
	"details": [
		{
			"name": null,
			"role": null,
			"organization": null,
			"notes": null
		}
	],
	"reason": null
}""",
	},
	{
		"code": "GOV_SUPERVISORY",
		"group": "Supervisory Board",
		"item": "Supervisory board full member list",
		"description": (
			"Trích xuất danh sách thành viên Ban Kiểm soát và thông tin từng thành viên."
		),
		"json_template": """{
	"value": {
		"total_members": null,
		"women_count": null,
		"men_count": null,
		"independent_count": null
	},
	"details": [
		{
			"name": null,
			"position": null,
			"gender": null,
			"is_independent": null,
			"date_of_birth": null,
			"appointment_date": null,
			"term_end": null,
			"education": null,
			"shares_owned": null,
			"notes": null
		}
	],
	"reason": null
}""",
	},
]
