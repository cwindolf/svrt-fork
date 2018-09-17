#!/bin/bash

#  svrt is the ``Synthetic Visual Reasoning Test'', an image generator
#  for evaluating classification performance of machine learning
#  systems, humans and primates.
#
#  Copyright (c) 2009 Idiap Research Institute, http://www.idiap.ch/
#  Written by Francois Fleuret <francois.fleuret@idiap.ch>
#
#  This file is part of svrt.
#
#  svrt is free software: you can redistribute it and/or modify it
#  under the terms of the GNU General Public License version 3 as
#  published by the Free Software Foundation.
#
#  svrt is distributed in the hope that it will be useful, but WITHOUT
#  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
#  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
#  License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with svrt.  If not, see <http://www.gnu.org/licenses/>.

nb_samples_to_save=100
nb_samples_for_training=1000

problem_list=$*

[[ ${problem_list} ]] || problem_list=$(echo {1..23})

set -e

make -j -k vision_test

for problem_number in ${problem_list}; do

    result_dir=./results_problem_${problem_number}/

    mkdir -p ${result_dir}

    ./vision_test \
        --problem_number=${problem_number} \
        --nb_train_samples=${nb_samples_to_save} \
        --result_path=${result_dir} \
        write-samples

    ./vision_test \
        --problem_number=${problem_number} \
        --nb_train_samples=${nb_samples_for_training} \
        --result_path=${result_dir} \
        --progress_bar=no \
        randomize-train adaboost compute-train-error compute-test-error | tee ${result_dir}/log

done
