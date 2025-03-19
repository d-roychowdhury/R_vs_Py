args <- commandArgs(trailingOnly = TRUE)
n_sim <- as.integer(args[1])
# Initialize an empty list to store the outputs
output_list <- list()

# Loop over the indices 1 to 200
for (i in 1:n_sim) {
  # Generate the file name dynamically
  file_name <- paste0("output_", i, ".rds")
  
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

## find the average of time elapsed
time_beta1_avg <- average_field(output_list, "time_beta1", n_sim)
time_beta2_avg <- average_field(output_list, "time_beta2", n_sim)

## the other req. averages
fr_mse_ols <- average_field (output_list, "err_ols", n_sim)
fr_mse_envlp <- average_field (output_list, "err_env", n_sim)
fr_mse_wbb1 <- average_field(output_list, "err_freq_beta1", n_sim)
fr_mse_wbb2<- average_field(output_list, "err_freq_beta2", n_sim)

bayes_mse_wbb2 <- average_field(output_list, "err_bayes_beta2", n_sim)

out_obj <- list("time_beta1_avg" =time_beta1_avg,
                "time_beta2_avg" = time_beta2_avg,
                "fr_mse_ols" = fr_mse_ols,
                "fr_mse_envlp"= fr_mse_envlp,
                "fr_mse_wbb1" = fr_mse_wbb1,
                "fr_mse_wbb2" = fr_mse_wbb2,
                "bayes_mse_wbb2" = bayes_mse_wbb2 )

#out_obj (original last line)
time_interest <- max(out_obj$time_beta1_avg, out_obj$time_beta2_avg) #(temp last line for timecheck.sh)
