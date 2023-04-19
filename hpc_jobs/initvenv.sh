module load python3/3.10.7
python3 -m venv venv
source venv/bin/activate
pip install -r requirements_imgx.txt
pip install -e .
