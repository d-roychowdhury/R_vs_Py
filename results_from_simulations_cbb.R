args <- commandArgs(trailingOnly = TRUE)
n_sim <- as.integer(args[1])
n <- as.integer(args[2])
p = as.integer(args[3])
r = as.integer(args[4])
u = as.integer(args[5])
# Initialize an empty list to store the outputs
output_list <- list()

# Loop over the indices 1 to n_sim
for (i in 1:n_sim) {
  # Generate the file name dynamically
  file_name <- paste0("output_", n, "_", p, "_", r, "_", u, "_", i,  ".rds")
  
  # Read the RDS file into a list
  output_list[[i]] <- readRDS(file_name)
}

average_field <- function(list_of_lists, field_name, n_sim){
  sum=0
  for(i in 1:n_sim){
    sum=sum + list_of_lists[[i]][[field_name]]
  }
  return(sum/n_sim)
}

median_field <- function(list_of_lists, field_name){

 med = median(list_of_lists[[i]][[field_name]])
 
  return(med)
}

## find the average of time elapsed
total_time_mean <- average_field (output_list, "total_time", n_sim)
total_time_median <- median_field(output_list, "total_time")

parallel_time_mean <- average_field (output_list, "time_parallel", n_sim)
parallel_time_median <- median_field(output_list, "time_parallel")
## the other req. averages
#fr_mse_ols <- average_field (output_list, "err_ols", n_sim)
fr_mse_envlp <- average_field (output_list, "err_freq_beta_env", n_sim)
fr_mse_cbb_method_1 <- average_field(output_list, "err_freq_beta1", n_sim)
fr_mse_cbb_method_2<- average_field(output_list, "err_freq_beta2", n_sim)

bayes_mse_cbb <- average_field(output_list, "err_bayes_beta_cbb", n_sim)

out_obj <- list(#"time_beta1_avg" =time_beta1_avg,
                #"time_beta2_avg" = time_beta2_avg,
                #"fr_mse_ols" = fr_mse_ols,
                "total_time_mean" = total_time_mean,
                "total_time_median" = total_time_median,
                "parallel_time_mean" = parallel_time_mean,
                "parallel_time_median" = parallel_time_median,
                "total_time_mean" = total_time,
                "fr_mse_envlp"= fr_mse_envlp,
                "fr_mse_cbb_method_1" = fr_mse_cbb_method_1,
                "fr_mse_cbb_method_2" = fr_mse_cbb_method_2,
                "bayes_mse_cbb" = bayes_mse_cbb )

print(out_obj)
saveRDS(out_obj, file =paste0("cbb_measures_", n, "_", p, "_", r, "_", u, "_with_nsim_", n_sim, ".rds"))
#out_obj (original last line)
#time_interest <- max(out_obj$time_beta1_avg, out_obj$time_beta2_avg) #(temp last line for timecheck.sh)
