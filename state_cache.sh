#!/bin/bash 


time_tag=`date +%Y_%m_%d_%H_%M_%S`
head_tag=`git rev-parse --short HEAD`
cache_path=trash/${time_tag}_${head_tag}

if [[ $# > 2 ]];
then
  cache_path=${1} 
fi

echo "Saving state at ${cache_path}"

cache_file() {
  if [ ! -d `dirname ${cache_path}/${1}` ];
  then
    mkdir -p `dirname ${cache_path}/${1}`
  fi
  cp ${1} ${cache_path}/${1}
}

json_file=`ls -rt *params*.json`
for i in ${json_file};
do
  new_json_file=${i}
done
echo "using ${new_json_file} as params file"

newer_file=`find -cnewer ${new_json_file} | grep -v trash | grep -E ".*\.(C|json|root|log|png|txt|pdf)"`
for i in ${newer_file};
do 
  cache_file ${i}
done
cache_file ${new_json_file}

npy_file=`ls -rt *.npy`
cache_file ${npy_file}
curve_file=`ls -rt *curve*`
cache_file ${curve_file}

git diff ${head_tag} > ${cache_path}/git.diff

echo "#!/bin/bash" > ${cache_path}/rebuild.sh
echo "git checkout ${head_tag}" >> ${cache_path}/rebuild.sh
echo "git checkout -b rebuild_${head_tag}" >> ${cache_path}/rebuild.sh
echo "git apply git.diff" >> ${cache_path}/rebuild.sh
