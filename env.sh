ONEAPI_ENV="$1"

echo $ONEAPI_ENV
source $ONEAPI_ENV/compiler/latest/env/vars.sh
source $ONEAPI_ENV/mkl/latest/env/vars.sh
source $ONEAPI_ENV/ccl/latest/env/vars.sh