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
INFERENCE_TOP_K: int = 5
INFERENCE_TEMPERATURE: float = float(os.getenv("INFERENCE_TEMPERATURE", "0"))

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
		"description": "danh gia cac rui ro (cac quy dinh, tac dong vat ly hoac cac tac dong chung) lien quan den bien doi khi hau va cac hanh dong da hoac se thuc hien de quan ly rui ro.",
	},
	{
		"code": "CC2",
		"group": "Climate Change",
		"item": "Financial implications of climate change",
		"description": "danh gia cac tac dong tai chinh hien tai (va tuong lai), tac dong kinh doanh va co hoi cua bien doi khi hau.",
	},
	{
		"code": "GHG1",
		"group": "GHG Emissions",
		"item": "Methodology for GHG emission calculation",
		"description": "Mo ta cac phuong phap su dung de tinh toan khi thai nha kinh.",
	},
	{
		"code": "GHG2",
		"group": "GHG Emissions",
		"item": "External verification of GHG emissions",
		"description": "Co su xac nhan boi yeu to ben ngoai ve luong phat thai khi nha kinh hay khong? neu co boi ai va tren co so gi",
	},
	{
		"code": "GHG3",
		"group": "GHG Emissions",
		"item": "Total GHG emissions disclosed",
		"description": "Luong phat thai khi nha kinh tinh bang don vi MtCO2e (he met tan CO2 thai ra).",
	},
	{
		"code": "GHG4",
		"group": "GHG Emissions",
		"item": "Disclosure of emissions by scope (Scope 1, 2, 3)",
		"description": "Viec cong bo lien quan den Pham vi 1, Pham vi 2, Pham vi 3 va lien quan truc tiep den phat thai khi nha kinh.",
	},
	{
		"code": "GHG5",
		"group": "GHG Emissions",
		"item": "Disclosure of emissions by source",
		"description": "Cong bo phat thai nha kinh dua tren nguon phat thai nao?",
	},
	{
		"code": "GHG6",
		"group": "GHG Emissions",
		"item": "Disclosure of emissions by facility or segment",
		"description": "Cong bo phat thai khi nha kinh dua tren co so vat chat hoac cap do nao?",
	},
	{
		"code": "GHG7",
		"group": "GHG Emissions",
		"item": "Historical comparison of emissions over time",
		"description": "So sanh luong phat thai khi nha kinh trong nam hien tai va nam truoc",
	},
	{
		"code": "EC1",
		"group": "Energy Consumption",
		"item": "Total energy consumed",
		"description": "Luong nang luong tieu thu.",
	},
	{
		"code": "EC2",
		"group": "Energy Consumption",
		"item": "Disclosure of energy consumption from renewable sources",
		"description": "Luong nang luong tieu thu co nguon goc tu nguon nang luong tai tao.",
	},
	{
		"code": "EC3",
		"group": "Energy Consumption",
		"item": "Disclosure of energy consumption by type, facility, or segment",
		"description": "Tiet lo dua tren loai khi thai, co so vat chat hoac cap do nao?",
	},
	{
		"code": "RC1",
		"group": "GHG Reduction",
		"item": "Plans or strategies to reduce GHG emissions",
		"description": "Giai thich chi tiet ve chien luoc va ke hoach giam thieu phat thai khi nha kinh.",
	},
	{
		"code": "RC2",
		"group": "GHG Reduction",
		"item": "Specific targets for GHG emission reduction",
		"description": "Muc tieu cu the ve luong giam thieu phat thai nha kinh",
	},
	{
		"code": "RC3",
		"group": "GHG Reduction",
		"item": "Reductions achieved to date",
		"description": "Luong giam phat thai toi da va cac chi phi hoac tiet kiem lien quan den giam phat khi thai nha kinh tinh den thoi diem bao cao.",
	},
	{
		"code": "RC4",
		"group": "GHG Reduction",
		"item": "Costs of future emissions factored into capital expenditure planning",
		"description": "Muc do chi phi lien quan den phat thai khi nha kinh trong tuong lai vi chi phi nay duoc bao gom trong ke hoach su dung von cua cong ty.",
	},
	{
		"code": "ACC1",
		"group": "GHG Accountability",
		"item": "Responsibility for climate change policy and action",
		"description": "Giai thich ai la Uy ban hoac Giam doc chiu trach nhiem ve cac chinh sach lien quan den bien doi khi hau.",
	},
	{
		"code": "ACC2",
		"group": "GHG Accountability",
		"item": "Consideration of climate-related goals in executive compensation",
		"description": "Giai thich co che xem xet khi dat duoc muc tieu cong ty lien quan den bien doi khi hau boi Uy ban hoac Hoi dong quan tri",
	},
]

# PROPER-VN items
PROPER_VN_STAGE1_ITEMS: list[dict[str, str]] = [
	{
		"code": "S1_VIOLATION",
		"group": "Stage 1: Regulatory Compliance",
		"item": "Serious environmental violations",
		"description": "Co bang chung hoac de cap den vi pham nghiem trong ve moi truong, bi xu phat hanh chinh nang, gay o nhiem, hoac bi dinh chi hoat dong vi ly do moi truong.",
		"score_if_true": "black",
	},
	{
		"code": "S1_MINOR_NC",
		"group": "Stage 1: Regulatory Compliance",
		"item": "Minor regulatory non-compliance",
		"description": "Co de cap den vi pham nho ve quy dinh moi truong, chua tuan thu day du cac tieu chuan phat thai, xu ly chat thai hoac quan ly moi truong theo quy dinh hien hanh.",
		"score_if_true": "red",
	},
	{
		"code": "S1_COMPLIANCE",
		"group": "Stage 1: Regulatory Compliance",
		"item": "Stated regulatory compliance",
		"description": "Cong ty tuyen bo tuan thu cac quy dinh ve moi truong, co giay phep moi truong, danh gia tac dong moi truong (DTM/EIA), hoac bao cao tuan thu quy chuan ky thuat moi truong.",
		"score_if_true": "compliant",
	},
]

PROPER_VN_STAGE2_ITEMS: list[dict[str, str]] = [
	{
		"code": "S2_ISO14001",
		"group": "Stage 2: Beyond-Compliance",
		"item": "ISO 14001 certification",
		"description": "Cong ty co chung nhan ISO 14001 hoac he thong quan ly moi truong tuong duong (EMS). Bao gom viec de cap den viec dat, duy tri hoac tai chung nhan ISO 14001.",
	},
	{
		"code": "S2_CARBON_DISC",
		"group": "Stage 2: Beyond-Compliance",
		"item": "Carbon emission disclosure",
		"description": "Cong ty cong bo luong phat thai carbon hoac khi nha kinh (GHG) cu the, bao gom Scope 1, Scope 2, hoac Scope 3, hoac tong phat thai tinh bang tan CO2 tuong duong.",
	},
	{
		"code": "S2_REDUCTION",
		"group": "Stage 2: Beyond-Compliance",
		"item": "Emission reduction targets or achievements",
		"description": "Cong ty co dat muc tieu giam phat thai khi nha kinh cu the (vi du: giam X% vao nam Y), hoac da dat duoc ket qua giam phat thai so voi nam goc, hoac cam ket Net Zero/Carbon Neutral.",
	},
	{
		"code": "S2_EFFICIENCY",
		"group": "Stage 2: Beyond-Compliance",
		"item": "Energy or resource efficiency initiatives",
		"description": "Cong ty thuc hien cac sang kien tiet kiem nang luong, su dung nang luong tai tao, tai che tai nguyen, giam tieu thu nuoc, hoac cac chuong trinh cai thien hieu suat su dung tai nguyen.",
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
			"Trich xuat danh sach co dong duoc de cap trong bao cao. "
			"Liet ke co dong lon (so huu tu 5% tro len) va co dong khac neu co. "
			"Voi moi co dong: ten, so co phan so huu, ty le so huu (%), "
			"loai co dong (ca nhan/to chuc/nha nuoc/nuoc ngoai). "
			"Tong hop: tong so co phieu dang luu hanh, co phieu quy, "
			"ty le so huu nuoc ngoai, ty le so huu nha nuoc, "
			"so co phan do to chuc nam giu, so co phan do ca nhan nam giu."
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
			"Trich xuat danh sach thanh vien Hoi dong quan tri (HDQT). "
			"Voi moi nguoi: ho ten, chuc vu HDQT, gioi tinh, ngay sinh, "
			"ngay bo nhiem, nhiem ky, trinh do hoc van, so co phan, ty le so huu, "
			"thanh vien doc lap hay khong. "
			"Neu la thanh vien doc lap thi ghi ro ly do/chung cu xac dinh doc lap. "
			"Khong trich xuat Ban Dieu hanh trong item nay."
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
			"Trich xuat danh sach Ban Dieu hanh va cac chuc danh dieu hanh "
			"(Tong Giam doc/CEO, Pho Tong Giam doc, Ke toan truong, chuc danh dieu hanh khac). "
			"Voi moi nguoi: ho ten, chuc vu dieu hanh, gioi tinh, ngay sinh, "
			"ngay bo nhiem, trinh do hoc van, co dong thoi la thanh vien HDQT hay khong. "
			"Neu la vai tro dieu hanh thi ghi ro ly do/chung cu tu chuc danh."
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
			"Trich xuat thong tin kiem toan doc lap cua cong ty: "
			"ten cong ty kiem toan doc lap, y kien kiem toan, "
			"kiem toan vien ky bao cao. "
			"Khong trich xuat thanh vien Ban Kiem soat hoac thanh vien Uy ban Kiem toan noi bo."
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
			"Trich xuat danh sach thanh vien Ban Kiem soat va thong tin tung thanh vien."
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
