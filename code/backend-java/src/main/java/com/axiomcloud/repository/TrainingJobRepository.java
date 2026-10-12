package com.axiomcloud.repository;

import com.axiomcloud.model.TrainingJobEntity;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface TrainingJobRepository extends JpaRepository<TrainingJobEntity, Integer> {

    Optional<TrainingJobEntity> findByJobId(String jobId);

    List<TrainingJobEntity> findAllByOrderByCreatedAtDesc();

    long countByStatus(String status);
}
