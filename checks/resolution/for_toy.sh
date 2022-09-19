mkdir results

for i in {1..100}; do
    bash run_all.sh
    cp final_params.json results/params${i}.json
done
