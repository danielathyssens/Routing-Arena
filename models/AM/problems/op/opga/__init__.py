
--boundary_.oOo._CBTOnwjY6WyXVK8lFePfxoJeXyOSW02s
Content-Length: 43
Content-Type: application/octet-stream
X-File-MD5: 943856ee0fc98242c6e80d8addad6ca1
X-File-Mtime: 1606137908
X-File-Path: /Thyssens/home/github_projects/RA/models/AM/problems/op/opga/README.md

# orienteering
GA for orienteering problem

--boundary_.oOo._CBTOnwjY6WyXVK8lFePfxoJeXyOSW02s
Content-Length: 5970
Content-Type: application/octet-stream
X-File-MD5: 1d7332750906b6631d27e580a36cd3bb
X-File-Mtime: 1606137908
X-File-Path: /Thyssens/home/github_projects/RA/models/AM/problems/op/opga/opevo.py

import sys
import random
import time
from . import oph

#fitness will take a set s and a set of weights and return a tuple containing the fitness and the best path
def fitness( chrom, s, start_point, end_point, tmax ):
    augs = []
    for i in range( len( s ) ):
        augs.append( ( s[ i ][0],
                       s[ i ][1],
                       s[ i ][2],
                       s[ i ][3], 
                       s[ i ][4] + chrom[ i ] ) )
    if debug:
        print ('fitness---------------------------------')
        print ('augs:')
        print (augs)
    #best = oph.ellinit_replacement( augs, start_point, end_point, tmax )
    ellset = oph.ell_sub( tmax, start_point, end_point, augs )
    #best = oph.initialize( ellset, start_point, end_point, tmax )[0]
    best = oph.init_replacement( ellset, start_point, end_point, tmax )[0]
    if debug:
        print ('best:')
        print (best)
        print ('best real reward:')
        print ([ x[3] for x in best ])
        print (len( s ))
        print ([ s[ x[3] - 2 ] for x in best[ 1:len( best ) - 1 ] ])
        print ([ s[ x[3] - 2 ][2] for x in best[ 1:len( best ) - 1 ] ])
        print (( sum( [ s[ x[3] - 2 ][2] for x in best[ 1:len( best ) - 1 ] ] ), best ))
    return ( sum( [ s[ x[3] - 2 ][2] for x in best[ 1:len( best ) - 1 ] ] ), best )

def crossover( c1, c2 ):
    assert( len( c1 ) == len( c2 ) )
    point = random.randrange( len( c1 ) )
    first = random.randrange( 2 )
    if( first ):
        return c1[:point] + c2[point:]
    else:
        return c2[:point] + c1[point:]

def mutate( chrom, mchance, msigma ):
    return [ x + random.gauss( 0, msigma ) if random.randrange( mchance ) == 0  else 
             x for x in chrom ]

def run_alg_f( f, tmax, N ):
    random.seed()
    cpoints = []
    an_unused_value = f.readline() # ignore first line of file
    for i in range( N ):
        cpoints.append( tuple( [ float( x ) for x in f.readline().split() ] ) )
    if debug:
        print ('N:            ', N)
    return run_alg(cpoints, tmax)

def run_alg(points, tmax, return_sol=False, verbose=True):
    cpoints = [tuple(p) + (i, 0) for i, p in enumerate(points)]
    start_point = cpoints.pop( 0 )
    end_point = cpoints.pop( 0 )
    assert( oph.distance( start_point, end_point ) < tmax )
    popsize = 10
    genlimit = 10
    kt = 5
    isigma = 10
    msigma = 7
    mchance = 2
    elitismn = 2
    if( debug ):
        print ('data set size:', len( cpoints ) + 2)
        print ('tmax:         ', tmax)
        print ('parameters:')
        print ('generations:     ', genlimit)
        print ('population size: ', popsize)
        print ('ktournament size:', kt)
        print ('mutation chance: ', mchance)
        print (str( elitismn ) + '-elitism')

    start_time = time.clock()
    #generate initial random population
    pop = []
    for i in range( popsize + elitismn ):
        chrom = []
        for j in range( len( cpoints ) ):
            chrom.append( random.gauss( 0, isigma ) )
        chrom = ( fitness( chrom, cpoints, start_point, end_point, tmax )[0], chrom )
        while( i - j > 0 and j < elitismn and chrom > pop[ i - 1 - j ] ):
            j += 1
        pop.insert( i - j, chrom )

    bestfit = 0
    for i in range( genlimit ):
        nextgen = []
        for j in range( popsize ):
            #select parents in k tournaments
            parents = sorted( random.sample( pop, kt ) )[ kt - 2: ] #optimize later
            #crossover and mutate
            offspring = mutate( crossover( parents[0][1], parents[1][1] ), mchance, msigma )
            offspring = ( fitness( offspring, cpoints, start_point, end_point, tmax )[0], offspring )
            if( offspring[0] > bestfit ):
                bestfit = offspring[0]
                if verbose:
                    print (bestfit)
            if( elitismn > 0 and offspring > pop[ popsize ] ):
                l = 0
                while( l < elitismn and offspring > pop[ popsize + l ] ):
                    l += 1
                pop.insert( popsize + l, offspring )
                nextgen.append( pop.pop( popsize ) )
            else:
                nextgen.append( offspring )
        pop = nextgen + pop[ popsize: ]

    bestchrom = sorted( pop )[ popsize + elitismn - 1 ] 
    end_time = time.clock()

    if verbose:
        print ('time:')
        print (end_time - start_time)
        print ('best fitness:')
        print (bestchrom[0])
        print ('best path:')
    best_path = fitness( bestchrom[1], cpoints, start_point, end_point, tmax )[1]
    if verbose:
        print ([ x[3] for x in best_path ])

        print ('their stuff:')
    stuff = oph.initialize( oph.ell_sub( tmax, start_point, end_point, cpoints )
    , start_point, end_point, tmax )[0]
    if verbose:
        print ('fitness:', sum( [ x[2] for x in stuff ] ))
        print ('my stuff:')
    stuff2 = oph.ellinit_replacement( cpoints, start_point, end_point, tmax )
    if verbose:
        print ('fitness:', sum( [ x[2] for x in stuff2 ] ))
        print ('checking correctness...')
    total_distance = ( oph.distance( start_point, cpoints[ best_path[ 1                    ][3] - 2 ] ) + 
                       oph.distance( end_point,   cpoints[ best_path[ len( best_path ) - 2 ][3] - 2 ] ) )
    for i in range( 1, len( best_path ) - 3 ):
        total_distance += oph.distance( cpoints[ best_path[ i     ][3] - 2 ], 
                                        cpoints[ best_path[ i + 1 ][3] - 2 ] )
    if verbose:
        print ('OK' if total_distance <= tmax else 'not OK')
        print ('tmax:          ', tmax)
        print ('total distance:', total_distance)
    if return_sol:
        return ( bestchrom[0], best_path, end_time - start_time )
    return ( bestchrom[0], end_time - start_time )

if( __name__ ==  '__main__' ):
    debug = True if 'd' in sys.argv else False
    run_alg( open( sys.argv[1] ), int( sys.argv[2] ), int( sys.argv[3] ) )
else:
    debug = False

--boundary_.oOo._CBTOnwjY6WyXVK8lFePfxoJeXyOSW02s
Content-Length: 1057
Content-Type: application/octet-stream
X-File-MD5: 477f436b8a55d6525bc5d2be8d6cecab
X-File-Mtime: 1606137908
X-File-Path: /Thyssens/home/github_projects/RA/models/AM/problems/op/opga/optest.py

import time
import opevo

files = [ 'test instances/set_64_1_15.txt' ]
tmaxs = [ range( 15,  80 + 1, 5 ) ]
Ns = [ 64 ]

test_runs = 30

assert( len( files ) == len( tmaxs ) and len( tmaxs ) == len( Ns ) )

for i in range( len( files ) ):
    f = open( files[ i ] )
    of = open( files[ i ][ :len( files[ i ] ) - 4 ] + '_results.dat', 'a' )

    of.write( time.asctime() + '\n' )
    of.write( 't avgfit avgtime bestfit\n' )
    for t in tmaxs[ i ]:
        fit_sum = float( 0 )
        time_sum = float( 0 )
        best_fit = 0
        for j in range( test_runs ):
            print('TEST %i/%i' % ( j + 1, test_runs ))
            f.seek( 0 ) 
            result = opevo.run_alg_f( f, t, Ns[ i ] )
            fit_sum += result[0]
            time_sum += result[1]
            best_fit = result[0] if result[0] > best_fit else best_fit
        #find avg fit, time, best fit then write to file
        of.write( ' '.join( [ str( x ) for x in [ t, fit_sum / test_runs, time_sum / test_runs,
            best_fit ] ] ) + '\n' )
    f.close()
    of.close()

--boundary_.oOo._CBTOnwjY6WyXVK8lFePfxoJeXyOSW02s
Content-Length: 9061
Content-Type: application/octet-stream
X-File-MD5: 1643348294dbe16c6a89e1dd84cbc390
X-File-Mtime: 1645797821
X-File-Path: /Thyssens/home/github_projects/RA/models/AM/problems/op/op_ortools.py

#!/usr/bin/problem python
# Th