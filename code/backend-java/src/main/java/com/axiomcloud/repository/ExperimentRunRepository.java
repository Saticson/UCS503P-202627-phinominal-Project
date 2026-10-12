package com.axiomcloud.repository;

import com.axiomcloud.model.ExperimentRunEntity;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface ExperimentRunRepository extends JpaRepository<ExperimentRunEntity, Integer> {

    Optional<ExperimentRunEntity> findByRunId(String runId);

    Optional<ExperimentRunEntity> findByJobId(String jobId);

    List<ExperimentRunEntity> findAllByOrderByStartedAtDesc();

    List<ExperimentRunEntity> findByStatusOrderByStartedAtDesc(String status);
}
