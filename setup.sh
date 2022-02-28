mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml

mkdir -p models/

wget https://ds-group22-adp.s3.eu-west-2.amazonaws.com/mlai/models/model_best_weights_anomaly_vae.h5 -o models/model_best_weights_anomaly_vae.h5
