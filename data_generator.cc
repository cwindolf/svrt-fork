/*
 *  svrt is the ``Synthetic Visual Reasoning Test'', an image
 *  generator for evaluating classification performance of machine
 *  learning systems, humans and primates.
 *
 *  Copyright (c) 2009 Idiap Research Institute, http://www.idiap.ch/
 *  Written by Francois Fleuret <francois.fleuret@idiap.ch>
 *
 *  This file is part of svrt.
 *
 *  svrt is free software: you can redistribute it and/or modify it
 *  under the terms of the GNU General Public License version 3 as
 *  published by the Free Software Foundation.
 *
 *  svrt is distributed in the hope that it will be useful, but
 *  WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with svrt.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <iostream>
#include <fstream>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

using namespace std;

#include "rgb_image.h"
#include "param_parser.h"
#include "global.h"

#include "vignette.h"
#include "shape.h"
#include "classifier.h"
#include "classifier_reader.h"
#include "naive_bayesian_classifier.h"
#include "boosted_classifier.h"
#include "error_rates.h"

#include "vision_problem_0.h"
#include "vision_problem_1.h"
#include "vision_problem_2.h"
#include "vision_problem_3.h"
#include "vision_problem_4.h"
#include "vision_problem_5.h"
#include "vision_problem_6.h"
#include "vision_problem_7.h"
#include "vision_problem_8.h"
#include "vision_problem_9.h"
#include "vision_problem_10.h"
#include "vision_problem_11.h"
#include "vision_problem_12.h"
#include "vision_problem_13.h"
#include "vision_problem_14.h"
#include "vision_problem_15.h"
#include "vision_problem_16.h"
#include "vision_problem_17.h"
#include "vision_problem_18.h"
#include "vision_problem_19.h"
#include "vision_problem_20.h"
#include "vision_problem_21.h"
#include "vision_problem_22.h"
#include "vision_problem_23.h"
#include "vision_problem_100.h"

//////////////////////////////////////////////////////////////////////

void check(bool condition, const char *message) {
  if(!condition) {
    cerr << message << endl;
    exit(1);
  }
}

int main(int argc, char **argv) {

  char buffer[buffer_size];
  char *new_argv[argc];
  int new_argc = 0;

  cout << "-- ARGUMENTS ---------------------------------------------------------" << endl;

  for(int i = 0; i < argc; i++)
    cout << (i > 0 ? "  " : "") << argv[i] << (i < argc - 1 ? " \\" : "")
         << endl;

  cout << "-- PARAMETERS --------------------------------------------------------" << endl;

  {
    ParamParser parser;
    global.init_parser(&parser);
    parser.parse_options(argc, argv, false, &new_argc, new_argv);
    global.read_parser(&parser);
    parser.print_all(&cout);
  }

  nice(global.niceness);
  srand48(global.random_seed);

  VignetteGenerator *generator;

  switch(global.problem_number) {
  case 0:
    generator = new VisionProblem_0();
    break;
  case 1:
    generator = new VisionProblem_1();
    break;
  case 2:
    generator = new VisionProblem_2();
    break;
  case 3:
    generator = new VisionProblem_3();
    break;
  case 4:
    generator = new VisionProblem_4();
    break;
  case 5:
    generator = new VisionProblem_5();
    break;
  case 6:
    generator = new VisionProblem_6();
    break;
  case 7:
    generator = new VisionProblem_7();
    break;
  case 8:
    generator = new VisionProblem_8();
    break;
  case 9:
    generator = new VisionProblem_9();
    break;
  case 10:
    generator = new VisionProblem_10();
    break;
  case 11:
    generator = new VisionProblem_11();
    break;
  case 12:
    generator = new VisionProblem_12();
    break;
  case 13:
    generator = new VisionProblem_13();
    break;
  case 14:
    generator = new VisionProblem_14();
    break;
  case 15:
    generator = new VisionProblem_15();
    break;
  case 16:
    generator = new VisionProblem_16();
    break;
  case 17:
    generator = new VisionProblem_17();
    break;
  case 18:
    generator = new VisionProblem_18();
    break;
  case 19:
    generator = new VisionProblem_19();
    break;
  case 20:
    generator = new VisionProblem_20();
    break;
  case 21:
    generator = new VisionProblem_21();
    break;
  case 22:
    generator = new VisionProblem_22();
    break;
  case 23:
    generator = new VisionProblem_23();
    break;
  case 100:
    generator = new VisionProblem_100();
    break;
  default:
    cerr << "Can not find problem "
         << global.problem_number
         << endl;
    exit(1);
  }

  generator->precompute();

  //////////////////////////////////////////////////////////////////////

  Vignette *train_samples;
  int *train_labels;

  train_samples = new Vignette[global.nb_train_samples];
  train_labels = new int[global.nb_train_samples];

  //////////////////////////////////////////////////////////////////////

  Classifier *classifier = 0;

  cout << "-- SAVING DATA ------------------------------------------------------" << endl;

  Vignette vignette;

  if (global.problem_number == 0) {
    // 0 does not care about labels.
    for(int k = 0; k < global.nb_train_samples; k++) {
      generator->generate(0, &vignette);
      sprintf(buffer, "%s/problem_0_sample_%01d.png", global.result_path, k);
      vignette.write_png(buffer, 1);
    }
  } else {
    for(int k = 0; k < global.nb_train_samples; k++) {
      for(int l = 0; l < 2; l++) {
        generator->generate(l, &vignette);
        sprintf(buffer, "%s/problem_%d_sample_%01d_%04d.png", global.result_path, global.problem_number, l, k);
        if (global.problem_number == 100) {
          vignette.write_rgb_png(buffer, 1);
        } else {
          vignette.write_png(buffer, 1);
        }
      }
    }
  }

  cout << "-- FINISHED ----------------------------------------------------------" << endl;

  delete classifier;
  delete[] train_labels;
  delete[] train_samples;
  delete generator;
}
