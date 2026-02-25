#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <pthread.h>

const int nbthreads = 20;

struct context {
  int start;
  int end;
  const int *data;
};

__attribute__((noinline)) static void task(int data) {
  struct timespec duration = {.tv_sec = 0, .tv_nsec = data};
  nanosleep(&duration, NULL);
}

void *thread_task(void *arg) {
  struct context *ctx = (struct context *)arg;
  const int *data = ctx->data;
  int start = ctx->start;
  int end = ctx->end;

  for (int i = start; i < end; ++i) {
    printf("Thread %lu processing data[%d] = %d\n", pthread_self(), i, data[i]);
    task(data[i]);
  }

  return NULL;
}

int min (int a, int b) { return (a < b) ? a : b; }

__attribute__((noinline)) static void work(int *data, int data_size) {
  pthread_t tid[nbthreads];
  struct context ctx[nbthreads];
  int chunk_size = data_size / nbthreads;

  for (int i = 0; i < nbthreads; ++i) {
    ctx[i].start = i * chunk_size;
    ctx[i].end = min((i + 1) * chunk_size, data_size);
    ctx[i].data = data;
    struct timeval time_start, time_end;
    gettimeofday((struct timeval *)&time_start, NULL);
    pthread_create(&tid[i], NULL, thread_task, &ctx[i]);
    gettimeofday((struct timeval *)&time_end, NULL);
    printf("Thread %lu created in %lf ms\n", tid[i],
           ((double)(time_end.tv_sec - time_start.tv_sec)) * 1000 +
               ((double)((time_end.tv_usec - time_start.tv_usec)) / 1000));
  }

  for (int i = 0; i < nbthreads; ++i)
    pthread_join(tid[i], NULL);
}

int main(int argc, char *argv[]) {
  if (argc != 2)
    return 1;

  int data_size = atoi(argv[1]);
  if (data_size == 0)
    return 2;

  int *data = malloc(sizeof(int) * data_size);

  // Create irregular workload.
  srand(0);
  struct timeval start, stop;
  for (int i = 0; i < data_size; ++i)
    data[i] = random() % 100000000l;

  // Process it.
  gettimeofday(&start, NULL);
  work(data, data_size);
  gettimeofday(&stop, NULL);
  printf("elapsed time: %lf ms\n",
         ((double)(stop.tv_sec - start.tv_sec)) * 1000 +
             ((double)((stop.tv_usec - start.tv_usec)) / 1000));

  return 0;
}
