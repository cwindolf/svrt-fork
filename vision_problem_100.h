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

#ifndef VISION_PROBLEM_100_H
#define VISION_PROBLEM_100_H

#include "vignette_generator.h"
#include "vision_problem_tools.h"

// This is vision problem 1, but with each shape in its own channel.
class VisionProblem_100 : public VignetteGenerator {
  static const int part_size = Vignette::width / 6;
  static const int hole_size = Vignette::width / 64;
public:
  VisionProblem_100();
  virtual void generate(int label, Vignette *vignette);
};

#endif
