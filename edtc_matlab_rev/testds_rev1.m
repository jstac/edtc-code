% Filename: testds_rev1.m
% Author: Tomohito Okabe
% Date: February 2013
% Corresponds to: Listing 4.3

% Clear all existing variables
clear all
% Close all figures
close all

% Define function h as a function handle
quadmap = @(x)4 * x * (1 - x);

% Set the initial state for T1 / the length of the trajectory 
x1 = 0.1;
N = 100;

% T1 holds tragectory from x1
T1 = ds_rev1(quadmap,x1,N);

% Set the initial state for T2
x2 = 0.2;

% T2 holds tragectory from x2
T2 = ds_rev1(quadmap,x2,N);
